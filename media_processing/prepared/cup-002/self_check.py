"""Geometric and visual checks of the recorded estimates; no image analysis."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

p = Path(__file__).parent
a = json.loads((p/'annotations.json').read_text(encoding='utf-8'))
K = a['camera']['intrinsic']
o = a['table']['origin_camera_m']
n = a['table']['up_normal_camera']
def project(v):
    return [K[0][0]*v[0]/v[2]+K[0][2], K[1][1]*v[1]/v[2]+K[1][2]]
bottom = project(o)
top = project([o[j]+.11*n[j] for j in range(3)])
height_px = math.dist(bottom,top)
centres = [s['eef']['position_camera_m'] for s in a['history']]
steps = [math.dist(centres[i],centres[i+1]) for i in range(20)]
known = sum(v is not None for row in a['depth']['values_m'] for v in row)
visible = sum(t['status']=='visible' for s in a['history'] for t in s['landmarks'])
inferred = sum(t['status']=='inferred' for s in a['history'] for t in s['landmarks'])
qa = {'projected_cup_bottom_px':bottom,'projected_cup_top_px':top,
      'projected_cup_height_px':height_px,'observed_mug_height_rough_px':295,
      'eef_step_m_min':min(steps),'eef_step_m_max':max(steps),
      'eef_step_m_mean':sum(steps)/len(steps),
      'depth_known_coarse':known,'depth_total_coarse':64*36,
      'hand_visible':visible,'hand_inferred':inferred,'hand_total':21*21,
      'checks_are_internal_only':True}
(p/'self-check.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

out = Image.new('RGB',(1920,1620),'white')
for j,i in enumerate((0,10,20)):
    with Image.open(p/'frames'/f'frame-{i+1:03d}.png') as im:
        panel=im.convert('RGB').resize((960,540))
    d=ImageDraw.Draw(panel)
    for t in a['history'][i]['landmarks']:
        if t['uv'] is None:
            continue
        x,y=t['uv'][0]*960,t['uv'][1]*540
        color='#ffff00' if t['status']=='visible' else '#ff00ff'
        d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        d.text((x+5,y),str(t['id']),fill=color,stroke_width=1,stroke_fill='black')
    d.text((10,10),f'sample {i}',fill='yellow',stroke_width=2,stroke_fill='black')
    out.paste(panel,(0,j*540))
out.save(p/'figures'/'hand-check-0-10-20.png')
print(json.dumps(qa,ensure_ascii=False))
