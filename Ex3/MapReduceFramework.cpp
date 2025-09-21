#include <atomic>
#include <semaphore.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "MapReduceFramework.h"
#include "Barrier.h"

constexpr uint64_t STAGE_SHIFT = 62;
constexpr uint64_t PROGRESS_MASK = 0x7FFFFFFF;
constexpr uint64_t TOTAL_SHIFT = 31;

typedef struct JobContext JobContext;
typedef struct Thread Thread;
void* runSingleThread(void* arg);

struct JobContext{
    const MapReduceClient* client;
    Thread* threadsArray;
    const InputVec* inputVector;
    std::vector<IntermediateVec>* intermediateVectors;
    OutputVec* outputVector;
    Barrier* barrier;
    pthread_mutex_t* shuffleMutex;
    pthread_mutex_t* reduceMutex;
    std::atomic<uint64_t>* stateCounter;
    std::atomic<int>* inputCounter;
    std::atomic<int>* intermediateCounter;
    std::atomic<int>* outputCounter;
    int threadsNumber;
    bool waitFlag;
};

struct Thread{
    JobContext* owner;
    pthread_t* pthread;
    IntermediateVec* intermediateVec;
    int threadId;
};

JobContext* createJobContext(const MapReduceClient& client,
                             const InputVec& inputVec, OutputVec& outputVec,
                             int multiThreadLevel) {
    auto* master = new JobContext ();
    master->client = &client;
    master->inputVector = &inputVec;
    master->intermediateVectors = new std::vector<IntermediateVec>();
    master->outputVector = &outputVec;
    master->threadsNumber = multiThreadLevel;
    master->threadsArray = new Thread[master->threadsNumber];
    master->barrier = new Barrier(master->threadsNumber);
    master->shuffleMutex = new pthread_mutex_t (PTHREAD_MUTEX_INITIALIZER);
    master->reduceMutex = new pthread_mutex_t (PTHREAD_MUTEX_INITIALIZER);
    master->stateCounter = new std::atomic<uint64_t>((long)inputVec.size()<<31);
    master->inputCounter = new std::atomic<int>(0);
    master->intermediateCounter = new std::atomic<int>(0);
    master->outputCounter = new std::atomic<int>(0);
    master->waitFlag = false;
    return master;
}

void destroyMutex(pthread_mutex_t* mutex) {
    if (pthread_mutex_destroy(mutex) != 0) {
        fprintf(stderr, "system error: failure to destroy mutex\n");
        exit(1);
    }
}

void lockMutex(pthread_mutex_t* mutex) {
    if (pthread_mutex_lock(mutex) != 0) {
        fprintf(stderr, "system error: failure to lock mutex\n");
        exit(1);
    }
}

void unlockMutex(pthread_mutex_t* mutex) {
    if (pthread_mutex_unlock(mutex) != 0) {
        fprintf(stderr, "system error: failure to unlock mutex\n");
        exit(1);
    }
}

void initializeThreadData(Thread& thread, JobContext* master, int threadId) {
    thread.owner = master;
    thread.pthread = new pthread_t();
    thread.intermediateVec = new IntermediateVec();
    thread.threadId = threadId;
}

void createThreads(JobContext* master) {
    for (int i = 0; i < master->threadsNumber; ++i) {
        initializeThreadData(master->threadsArray[i], master, i);
        int result = pthread_create(master->threadsArray[i].pthread, nullptr,
                                    runSingleThread, &(master->threadsArray[i]));
        if (result != 0) {
            fprintf(stderr, "system error: failure to create pthread\n");
            exit(1);
        }
    }
}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
    JobContext* master = createJobContext(client, inputVec, outputVec, multiThreadLevel);
    //MAYBE FIX THIS FUNCTION, UPDATE IF NECESSARY AFTER FIRST RUN.
    createThreads(master);
    return static_cast<JobHandle>(master);
}

void waitForJob(JobHandle job) {
    auto* master = static_cast<JobContext*>(job);
    if (master->waitFlag){
        return;
    }
    for (int i = 0; i < master->threadsNumber; ++i){
        int result = pthread_join(*(master->threadsArray[i].pthread), nullptr);
        if (result != 0){
            fprintf(stderr, "system error: failure to join pthreads\n");
            exit(1);
        }
    }
    master->waitFlag = true;
}

void getJobState(JobHandle job, JobState* state) {
    auto* master = static_cast<JobContext*>(job);
    uint64_t value = master->stateCounter->load();
    state->stage = static_cast<stage_t>(value >> STAGE_SHIFT);
    uint64_t stateProgress = value & PROGRESS_MASK;
    uint64_t totalProgress = (value >> TOTAL_SHIFT) & PROGRESS_MASK;
    if (totalProgress != 0){
        state->percentage = (static_cast<float>(stateProgress) / static_cast<float>(totalProgress)) * 100.0f;
    }
    else{
        state->percentage = 0.0f;
    }
}

void closeJobHandle(JobHandle job) {
    auto *master = static_cast<JobContext *>(job);
    if (master == nullptr) {
        return;
    }
    waitForJob(job);
    delete master->inputCounter;
    delete master->intermediateCounter;
    delete master->outputCounter;
    delete master->stateCounter;
    for (int i = 0; i < master->threadsNumber; ++i) {
        delete master->threadsArray[i].pthread;
        delete master->threadsArray[i].intermediateVec;
    }
    delete[] master->threadsArray;
    destroyMutex(master->shuffleMutex);
    destroyMutex(master->reduceMutex);
    delete master->reduceMutex;
    delete master->shuffleMutex;
    delete master->intermediateVectors;
    delete master->barrier;
    delete master;
}

void emit2(K2* key, V2* value, void* context) {
    auto* intermediateContext = static_cast<Thread*>(context);
    intermediateContext->intermediateVec->emplace_back(key, value);
    (*intermediateContext->owner->intermediateCounter)++;
}

void emit3(K3* key, V3* value, void* context) {
    auto job = (JobContext*) context;

    lockMutex(job->shuffleMutex);

    job->outputVector->emplace_back(key, value);
    (*job->outputCounter)++;

    unlockMutex(job->shuffleMutex);
}

bool comparePairs(const IntermediatePair& lhs, const IntermediatePair& rhs) {
    return *lhs.first < *rhs.first;
}

void mapAndSort(Thread* thread) {
    auto job = thread->owner;
    auto& stateCounter = *(job->stateCounter);
    auto& inputCounter = *(job->inputCounter);
    auto& inputVector = *(job->inputVector);
    auto client = job->client;
    auto& intermediateVec = *(thread->intermediateVec);
    stateCounter |= ((long)MAP_STAGE << 62);
    int num = inputCounter++;
    while (num < static_cast<int>(inputVector.size())) {
        InputPair pair = inputVector.at(num);
        client->map(pair.first, pair.second, thread);
        stateCounter++;
        num = inputCounter++;
    }
    std::sort(intermediateVec.begin(), intermediateVec.end(), comparePairs);
}

void updateIntermediates(JobContext* job, std::vector<int>& hasData) {
    for (int i = 0; i < job->threadsNumber; i++) {
        if (!job->threadsArray[i].intermediateVec->empty()) {
            hasData.push_back(i);
        }
    }
}

K2* getMaxKey(JobContext* job, const std::vector<int>& hasData) {
    K2* maxKey = job->threadsArray[hasData[0]].intermediateVec->back().first;
    for (int i : hasData) {
        if (*maxKey < *(job->threadsArray[i].intermediateVec->back().first)) {
            maxKey = job->threadsArray[i].intermediateVec->back().first;
        }
    }
    return maxKey;
}

void updatePairs(JobContext* job, K2* optional, std::vector<int>& hasData,
                 IntermediateVec& intermediateVectors) {
    std::vector<int> indicesToRemove;
    for (int i : hasData) {
        while (!(*(job->threadsArray[i].intermediateVec->back().first) < *optional)) {
            intermediateVectors.push_back(job->threadsArray[i].intermediateVec->back());
            job->threadsArray[i].intermediateVec->pop_back();
            (*job->stateCounter)++;
            if (job->threadsArray[i].intermediateVec->empty()) {
                indicesToRemove.push_back(i);
                break;
            }
        }
    }

    for (int i : indicesToRemove) {
        auto it = std::find(hasData.begin(), hasData.end(), i);
        if (it != hasData.end()) {
            hasData.erase(it);
        }
    }
}

void shuffle(JobContext* job) {
    std::vector<int> hasData;
    updateIntermediates(job, hasData);

    IntermediateVec intermediateVectors;

    while (!hasData.empty()) {
        K2* optional = getMaxKey(job, hasData);
        intermediateVectors.clear();
        updatePairs(job, optional, hasData, intermediateVectors);
        job->intermediateVectors->push_back(intermediateVectors);
    }
}

void* runSingleThread(void* arg) {
    auto* thread = static_cast<Thread*>(arg);
    mapAndSort(thread);
    thread->owner->barrier->barrier();
    if(thread->threadId == 0){
        thread->owner->stateCounter->exchange(thread->owner->intermediateCounter->load()<<31 |
                                              ((uint64_t) SHUFFLE_STAGE<<62));
        shuffle(thread->owner);
        thread->owner->stateCounter->exchange(thread->owner->intermediateVectors->size()<<31 |
                                              (uint64_t) REDUCE_STAGE<<62);
    }
    thread->owner->barrier->barrier();
    IntermediateVec intermediateVector;
    while (!(*(thread->owner->intermediateVectors)).empty()) {
        lockMutex(thread->owner->reduceMutex);
        if ((*(thread->owner->intermediateVectors)).empty()) {
            unlockMutex(thread->owner->reduceMutex);
            break;
        }
        intermediateVector = thread->owner->intermediateVectors->back();
        thread->owner->intermediateVectors->pop_back();
        unlockMutex(thread->owner->reduceMutex);
        thread->owner->client->reduce(&intermediateVector, thread->owner);
        (*thread->owner->stateCounter)++;

    }
    return nullptr;
}


