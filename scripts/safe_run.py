"""Run a command under a hard memory cap and a timeout, so a runaway test can't freeze the machine.

python safe_run.py --mem-gb 8 --timeout 600 -- <command> [args...]

The command (and every child it starts) is placed in a Windows Job Object with a job-wide committed-memory
limit; Windows fails allocations past it and the job is killed. Prints the peak memory and exit code.
"""
import argparse
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

k32 = ctypes.WinDLL('kernel32', use_last_error=True)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                   'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class BASIC(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD)]


class EXTENDED(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO_COUNTERS), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]


JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.c_void_p]
k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]

parser = argparse.ArgumentParser()
parser.add_argument('--mem-gb', type=float, default=8)
parser.add_argument('--timeout', type=float, default=600)
parser.add_argument('cmd', nargs=argparse.REMAINDER)
args = parser.parse_args()
cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd

job = k32.CreateJobObjectW(None, None)
info = EXTENDED()
info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
info.JobMemoryLimit = int(args.mem_gb * 2**30)
if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
    sys.exit(f'SetInformationJobObject failed: {ctypes.get_last_error()}')

CREATE_SUSPENDED = 0x4
proc = subprocess.Popen(cmd, creationflags=CREATE_SUSPENDED)
if not k32.AssignProcessToJobObject(job, int(proc._handle)):
    proc.kill()
    sys.exit(f'AssignProcessToJobObject failed: {ctypes.get_last_error()}')
# Resume the main thread (Popen started it suspended so nothing ran outside the job).
ntdll = ctypes.WinDLL('ntdll')
ntdll.NtResumeProcess(int(proc._handle))

started = time.time()
status = 'ok'
while proc.poll() is None:
    if time.time() - started > args.timeout:
        k32.TerminateJobObject(job, 1)
        status = 'TIMEOUT'
        break
    time.sleep(0.5)
code = proc.wait()
k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info), None)
peak = info.PeakJobMemoryUsed / 2**30
if status == 'ok' and code != 0 and peak > args.mem_gb * 0.95:
    status = 'MEMORY CAP HIT'
print(f'[safe_run] {status}: exit {code}, peak memory {peak:.2f} GB (cap {args.mem_gb} GB), '
      f'{time.time() - started:.0f}s', file=sys.stderr, flush=True)
sys.exit(code)
