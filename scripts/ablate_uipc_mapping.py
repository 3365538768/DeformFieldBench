"""Separate installed-wheel mapping and mesh effects at the SAME E/load/geometry."""
import argparse
import json
from pathlib import Path
import importlib.metadata

from calibrate_uipc_beam import run_case


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='docs/UIPC_MAPPING_GRID_ABLATION.json');a=p.parse_args()
    native=json.loads(Path('docs/UIPC_BEAM_CALIBRATION.json').read_text())['cases'][1]
    corrected=json.loads(Path('docs/UIPC_BEAM_CORRECTED_CALIBRATION.json').read_text())['cases'][1]
    result={'status':'running','engine':'libuipc','version':importlib.metadata.version('pyuipc'),
            'cases':{'native_n20':native,'snk1_physical_n80':corrected},
            'note':'Fixed E=1e6 Pa, nu=.3, force=.001 N, same beam and exact clamp. Factorial isolates mapping versus mesh; no global cross-engine conversion claimed.'}
    output=Path(a.output)
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
    save()
    for resolution,mapping in [(20,'snk1_physical'),(80,'native')]:
        result['cases'][f'{mapping}_n{resolution}']=run_case(Path('data/calibration/uipc_mapping_ablation')/f'{mapping}_n{resolution}',resolution,1e6,mapping)
        save()
    cases=result['cases']
    result['mapping_deflection_ratio_at_n20']=cases['snk1_physical_n20']['tip_deflection_m']/cases['native_n20']['tip_deflection_m']
    result['mapping_deflection_ratio_at_n80']=cases['snk1_physical_n80']['tip_deflection_m']/cases['native_n80']['tip_deflection_m']
    result['status']='complete';save()
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
