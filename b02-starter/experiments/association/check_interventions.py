"""Synthetic checks for the actual intervention boundaries."""
from collections import defaultdict
from copy import deepcopy
from math import floor
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(project),str(project/'experiments/point_scorer_expanded')]
from b02.algorithm import DEFAULT_CONFIG, Track
from assignment import assign
from interventions import Tracker, priority_assign


def main():
    tracks={1:Track((0,0,0),1,hits=3),2:Track((0,0,0),1,hits=3)}
    grid=defaultdict(list,{(0,0,0):[0]})
    args=([1,2],tracks,2,[(0,0,0)],grid)
    original=assign(*args,set(),DEFAULT_CONFIG,scores=[.9],score_weight=.50)
    equivalent=priority_assign(*args,set(),DEFAULT_CONFIG,scores=[.9],strong_hits={1:6,2:6})
    assert original==equivalent
    favored=priority_assign(*args,set(),DEFAULT_CONFIG,scores=[.9],strong_hits={1:0,2:6})
    assert favored=={2:0} and len(set(favored.values()))==len(favored)

    point={'Time':1,'CircleID':1,'PntID':1,'Azimuth':1.5*.025,'Elevation':0.,'Distance':0.,'Amp':1.,'SNR':1.}
    initial={'scan':1,'next_id':2,'tracks':{1:Track((0,0,0),1,velocity=(1,0,0),hits=3)}}
    control,soft=Tracker('control'),Tracker('confidence_velocity')
    control.state,soft.state=deepcopy(initial),deepcopy(initial)
    control([point],[.1]);soft([point],[.1])
    a,b=control.state['tracks'][1],soft.state['tracks'][1]
    assert abs(b.velocity[0]-1)<abs(a.velocity[0]-1)
    assert (a.position,a.hits,a.last_scan)==(b.position,b.hits,b.last_scan)
    control,soft=Tracker('control'),Tracker('confidence_velocity')
    control.state,soft.state=deepcopy(initial),deepcopy(initial)
    assert control([point],[1.])==soft([point],[1.])
    assert control.state['tracks']==soft.state['tracks']
    print('Passed: equal-evidence reproduces association; stronger evidence wins one-to-one competition; low-score velocity influence decreases; high-score state remains exact.')


if __name__=='__main__':
    main()
