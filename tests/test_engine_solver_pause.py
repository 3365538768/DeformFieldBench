import sys
from pathlib import Path
import subprocess

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import engine_fixture as fixture


def fake_process(monkeypatch, finish=11., code=0):
    clock=[0.]
    class Process:
        returncode=None
        terminated=False
        def poll(self):
            if clock[0]>=finish and self.returncode is None:self.returncode=code
            return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self,timeout=None):return self.returncode
    proc=Process()
    monkeypatch.setattr(fixture.subprocess,'Popen',lambda *a,**kw:proc)
    monkeypatch.setattr(fixture.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(fixture.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    return proc,clock


def test_paused_wall_time_does_not_consume_solver_budget(monkeypatch,tmp_path):
    process,clock=fake_process(monkeypatch)
    monkeypatch.setattr(fixture,'is_paused',lambda output:.5<=clock[0]<10.5)
    fixture.run_controlled_solver(['owned-solver'],None,tmp_path,2.,tmp_path)
    assert clock[0]==11. and not process.terminated


def test_active_timeout_terminates_only_spawned_solver(monkeypatch,tmp_path):
    process,_=fake_process(monkeypatch)
    monkeypatch.setattr(fixture,'is_paused',lambda output:False)
    with pytest.raises(subprocess.TimeoutExpired):
        fixture.run_controlled_solver(['owned-solver'],None,tmp_path,2.,tmp_path)
    assert process.terminated


def test_failed_solver_is_not_marked_successful(monkeypatch,tmp_path):
    process,_=fake_process(monkeypatch,finish=0,code=7)
    with pytest.raises(subprocess.CalledProcessError) as error:
        fixture.run_controlled_solver(['owned-solver'],None,tmp_path,2.,tmp_path)
    assert error.value.returncode==7 and not process.terminated
