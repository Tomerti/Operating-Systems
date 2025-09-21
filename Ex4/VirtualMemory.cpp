#include "PhysicalMemory.h"

struct FrameInfo {
    int frame;
    int parentFrame;
    int offset;
};

struct MaxPageInfo {
    int frame;
    int distance;
    int pageNumber;
    int parentFrame;
    int offset;
};

uint64_t traverseVirtualAddress(uint64_t virtualAddress);
void initializeFrame(int frameNumber);
void addPageToTree(int pageTableAddresses[TABLES_DEPTH], int depth, int pageIndex, int currentFrame,
                   FrameInfo& frameInfo, int ancestor, int offset, int& frameIndex, int replacedPage, MaxPageInfo& maxPageInfo);
int selectFrameForPage(int pageTableAddresses[TABLES_DEPTH], int swappedPage);

template <typename T>
T absolute(T value) {
    if (value < 0) {
        return value * -1;
    }
    return value;
}

template <typename T>
T min(T x, T y) {
    if (x < y) {
        return x;
    }
    return y;
}

void VMinitialize() {
    initializeFrame(0);
}

int VMread(uint64_t virtualAddress, word_t* value) {
    if (virtualAddress < VIRTUAL_MEMORY_SIZE) {
        uint64_t physicalAddress = traverseVirtualAddress(virtualAddress);
        if ((int)physicalAddress != -1) {
            uint64_t offset = virtualAddress % (1ULL << OFFSET_WIDTH);
            PMread(physicalAddress * PAGE_SIZE + offset, value);
            return 1;
        }
    }
    return 0;
}

int VMwrite(uint64_t virtualAddress, word_t value) {
    if (virtualAddress < VIRTUAL_MEMORY_SIZE) {
        uint64_t physicalAddress = traverseVirtualAddress(virtualAddress);
        if ((int)physicalAddress != -1) {
            uint64_t offset = virtualAddress % (1ULL << OFFSET_WIDTH);
            PMwrite(physicalAddress * PAGE_SIZE + offset, value);
            return 1;
        }
    }
    return 0;
}

void updatePageTableAddresses(int pageTableAddresses[], int depth, word_t tempAddress) {
    pageTableAddresses[depth + 1] = tempAddress;
}

uint64_t calculateOffset(uint64_t virtualAddress, int depth) {
    uint64_t offsetMask = (1 << OFFSET_WIDTH) - 1;
    return (virtualAddress >> ((TABLES_DEPTH - depth) * OFFSET_WIDTH)) & offsetMask;
}

word_t handleEmptyAddress(uint64_t physicalAddress, uint64_t offset, uint64_t virtualAddress, int depth, int pageTableAddresses[]) {
    word_t tempAddress = selectFrameForPage(pageTableAddresses, virtualAddress >> OFFSET_WIDTH);
    if (tempAddress == -1) {
        return -1;
    }
    PMwrite(physicalAddress * PAGE_SIZE + offset, tempAddress);
    if (depth == TABLES_DEPTH - 1) {
        PMrestore(tempAddress, virtualAddress >> OFFSET_WIDTH);
    } else {
        initializeFrame(tempAddress);
    }
    return tempAddress;
}

uint64_t traverseVirtualAddress(uint64_t virtualAddress) {
    uint64_t offset;
    uint64_t physicalAddress = 0;
    word_t optional;
    int pageTableAddresses[TABLES_DEPTH + 1];
    for(int i=0;i<TABLES_DEPTH;i++){
        pageTableAddresses[i] = 0;
    }
    int depth = 0;

    while (depth < TABLES_DEPTH) {
        offset = calculateOffset(virtualAddress, depth);
        PMread(physicalAddress * PAGE_SIZE + offset, &optional);
        updatePageTableAddresses(pageTableAddresses, depth, optional);

        if (optional == 0) {
            optional = handleEmptyAddress(physicalAddress, offset, virtualAddress, depth, pageTableAddresses);
            if (optional == -1) {
                return -1;
            }
        }
        physicalAddress = optional;
        updatePageTableAddresses(pageTableAddresses, depth, optional);
        ++depth;
    }
    return physicalAddress;
}

void initializeFrame(int frameNumber) {
    for (int i = 0; i < PAGE_SIZE; ++i) {
        PMwrite(frameNumber * PAGE_SIZE + i, 0);
    }
}

void updateMaxPageInfo(int replacedPage, int pageIndex, int currentFrame, int ancestor, int offset, MaxPageInfo& maxPageInfo) {
    int distance = min((int)NUM_PAGES - absolute(replacedPage - pageIndex), absolute(replacedPage - pageIndex));
    if (distance > maxPageInfo.distance) {
        maxPageInfo.distance = distance;
        maxPageInfo.frame = currentFrame;
        maxPageInfo.pageNumber = pageIndex;
        maxPageInfo.parentFrame = ancestor;
        maxPageInfo.offset = offset;
    }
}

void addPageToTree(int pageTable[TABLES_DEPTH], int depth, int pageIndex, int currentFrame,
                   FrameInfo& frameInfo, int ancestor, int offset, int& frameIndex, int replacedPage, MaxPageInfo& maxPageInfo) {
    ++frameIndex;

    if (depth == TABLES_DEPTH) {
        updateMaxPageInfo(replacedPage, pageIndex, currentFrame, ancestor, offset, maxPageInfo);
        return;
    }

    bool isEmptyFrame = true;
    int address = 0;
    int i = 0;

    while (i < PAGE_SIZE) {
        PMread(currentFrame * PAGE_SIZE + i, &address);
        if (address != 0) {
            isEmptyFrame = false;
            int updatedIndex = (pageIndex << OFFSET_WIDTH) + i;
            addPageToTree(pageTable, depth + 1, updatedIndex, address,
                          frameInfo, currentFrame, i, frameIndex, replacedPage, maxPageInfo);
        }
        ++i;
    }

    if (isEmptyFrame) {
        i = 0;
        while (i < TABLES_DEPTH) {
            if (pageTable[i] == currentFrame) {
                return;
            }
            ++i;
        }
        frameInfo.frame = currentFrame;
        frameInfo.parentFrame = ancestor;
        frameInfo.offset = offset;
    }
}

FrameInfo initializeFrameInfo() {
    FrameInfo frameInfo{};
    frameInfo.frame = 0;
    frameInfo.parentFrame = 0;
    frameInfo.offset = 0;
    return frameInfo;
}

MaxPageInfo initializeMaxPageInfo() {
    MaxPageInfo maxPageInfo{};
    maxPageInfo.frame = 0;
    maxPageInfo.distance = -1;
    maxPageInfo.pageNumber = 0;
    maxPageInfo.parentFrame = 0;
    maxPageInfo.offset = 0;
    return maxPageInfo;
}

void clearFrame(FrameInfo& frameInfo) {
    PMwrite(frameInfo.parentFrame * PAGE_SIZE + frameInfo.offset, 0);
}

void clearMaxPageFrame(MaxPageInfo& maxPageInfo) {
    PMwrite(maxPageInfo.parentFrame * PAGE_SIZE + maxPageInfo.offset, 0);
}

int selectFrameForPage(int pageTable[TABLES_DEPTH], int replacedPage) {
    FrameInfo frameInfo = initializeFrameInfo();
    MaxPageInfo maxPageInfo = initializeMaxPageInfo();
    int frameIndex = 0;

    addPageToTree(pageTable, 0, 0, 0, frameInfo, 0, 0, frameIndex, replacedPage, maxPageInfo);

    if (frameInfo.frame != 0) {
        clearFrame(frameInfo);
        return frameInfo.frame;
    }

    if (frameIndex < NUM_FRAMES) {
        return frameIndex;
    }

    if (maxPageInfo.frame != 0) {
        PMevict(maxPageInfo.frame, maxPageInfo.pageNumber);
        clearMaxPageFrame(maxPageInfo);
        return maxPageInfo.frame;
    }

    return -1;
}