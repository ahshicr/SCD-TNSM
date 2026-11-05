"""Generate manuscript numerical macros with an exact source-cell registry."""
import json
import math


def write_macros(target,registry,entries):
    lines=['% Generated from verified full-matrix results. Do not edit numerical values.']
    records=[]
    seen=set()
    for name,value,digits,source in entries:
        if not name.isascii() or not name.isalpha() or name in seen:
            raise ValueError('LaTeX numerical macro names must be unique ASCII words')
        seen.add(name)
        if value is None or not math.isfinite(float(value)):
            display=r'\text{not available}'
            raw=None
        else:
            raw=float(value)
            display=f'{raw:.{digits}f}' if digits is not None else f'{raw:.6g}'
            if 'e' in display:
                mantissa,exponent=display.split('e')
                display=mantissa+r'\times10^{'+str(int(exponent))+'}'
            elif float(display)==0:
                display=display.lstrip('-')
        lines.append('\\newcommand{\\'+name+'}{\\ensuremath{'+display+'}}')
        records.append(dict(macro=name,value=raw,display=display,source=source))
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    registry.write_text(json.dumps(dict(complete=True,macros=records),indent=2),encoding='utf-8')
