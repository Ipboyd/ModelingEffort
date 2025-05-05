#include "malloc.h"

void* AlignedMalloc(size_t size,int byteAlign)
{
	void *mallocPtr = malloc(size + byteAlign + sizeof(void*));
	size_t ptrInt = (size_t)mallocPtr;

	ptrInt = (ptrInt + byteAlign + sizeof(void*)) / byteAlign * byteAlign;
	*(((void**)ptrInt) - 1) = mallocPtr;

    return (void*)ptrInt;
}

void AlignedFree(void *ptr)
{
	free(*(((void**)ptr) - 1));
}
