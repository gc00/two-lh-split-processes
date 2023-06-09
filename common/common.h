/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include <link.h>
#include <stdio.h>
#include <string.h>

#include <asm/prctl.h>
#include <linux/limits.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#include "mpi_lh/lower_half_mpi_if.h"

typedef char* VA;  /* VA = virtual address */

// Based on the entries in /proc/<pid>/stat as described in `man 5 proc`
enum Procstat_t
{
  PID = 1,
  COMM,   // 2
  STATE,  // 3
  PPID,   // 4
  NUM_THREADS = 19,
  STARTSTACK = 27,
};

#define PAGE_SIZE 0x1000LL

// FIXME: 0x1000 is one page; Use sysconf(PAGESIZE) instead.
#define ROUND_DOWN(x) ((unsigned long long)(x) \
                      & ~(unsigned long long)(PAGE_SIZE - 1))
#define ROUND_UP(x)  (((unsigned long long)(x) + PAGE_SIZE - 1) & \
                      ~(PAGE_SIZE - 1))
#define PAGE_OFFSET(x)  ((x) & (PAGE_SIZE - 1))

// TODO: This is very x86-64 specific; support other architectures??
#define eax rax
#define ebx rbx
#define ecx rcx
#define edx rax
#define ebp rbp
#define esi rsi
#define edi rdi
#define esp rsp
#define CLEAN_FOR_64_BIT_HELPER(args ...) # args
#define CLEAN_FOR_64_BIT(args ...)        CLEAN_FOR_64_BIT_HELPER(args)

typedef struct __LowerHalfInfo
{
  void *lhSbrk;
  void *lhMmap;
  void *lhMunmap;
  void *lhDlsymMPI;
  unsigned long lhFsAddr;
  void *lhMmapListFptr;
  void *uhEndofHeapFptr;
  void *lhDeviceHeap;
} LowerHalfInfo_t;

typedef struct __UpperHalfInfo
{
  void *uhEndofHeap;
  void *lhPagesRegion;
} UpperHalfInfo_t;

typedef struct __MmapInfo
{
  void *addr;
  size_t len;
} MmapInfo_t;

extern LowerHalfInfo_t lhInfo;
extern UpperHalfInfo_t uhInfo;

#ifdef __cplusplus
extern "C" {
#endif
void* lhDlsymMPI(MPI_Fncs_t type);
#ifdef __cplusplus
}
#endif

typedef void* (*LhDlsymMPI_t)(MPI_Fncs_t type);
#endif //ifndef COMMON_H
