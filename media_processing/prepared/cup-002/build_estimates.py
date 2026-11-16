"""Second direct visual estimate from the frozen 21-frame history.

Geometry code only interpolates the manually chosen camera, table, silhouettes,
current hand anchors, and the per-frame motion estimates in annotations.json.
"""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

p = Path(__file__).parent
q = p/'annotations.json'
a = json.loads((p.parent/'cup-001'/'annotations.json').read_text(encoding='utf-8'))

# Mug rim around y=353 and base around y=647: these camera/table numbers reproject
# a vertical 0.11 m segment to about 293 pixels at the visible mug centre.
fx = fy = 1360.0
cx, cy = 960.0, 540.0
origin = [-0.097, 0.040, 0.507]
up = [0.0, -0.95, -0.31]
a['camera']['intrinsic'] = [[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]]
a['camera']['notes'] = ('Вторая визуальная оценка K: fx=fy=1360 px, центр в центре кадра. '
                       'Проверка с высотой кружки 0,11 м даёт около 293 px между дном и верхом; '
                       'видимая высота около 295 px. Реальный K и дисторсия неизвестны.')
a['table']['origin_camera_m'] = origin
a['table']['up_normal_camera'] = up
a['table']['notes'] = ('Начало под видимым основанием кружки. Нормаль получена из направления '
                      'удаления столешницы и согласована с высотой кружки; это оценка, не калибровка.')

# Rasterize only polygons manually identified in the current RGB. A semantic mask
# locates where to evaluate the visual depth estimate; it does not recognize pixels.
mask = Image.new('L',(64,36),0)
d = ImageDraw.Draw(mask)
ids = {'table':1,'headphones':2,'mug':3,'hand':4}
for obj in a['objects']:
    vertices = [(u*64-.5,v*36-.5) for u,v in obj['polygon_uv']]
    d.polygon(vertices,fill=ids[obj['id']])

n = [v/math.sqrt(sum(t*t for t in up)) for v in up]
plane = sum(n[k]*origin[k] for k in range(3))
depth = [[None]*64 for _ in range(36)]
for row in range(36):
    for col in range(64):
        category = mask.getpixel((col,row))
        if category == 0:
            continue
        x = (col+.5)*1920/64
        y = (row+.5)*1080/36
        ray = [(x-cx)/fx,(y-cy)/fy,1.0]
        if category == 1:  # tabletop: intersection of the visually estimated plane
            den = sum(n[k]*ray[k] for k in range(3))
            z = plane/den if abs(den)>1e-8 else math.nan
            if .25 < z < 1.5:
                depth[row][col] = round(z,4)
        elif category == 2:  # headphones sit farther back than the mug
            depth[row][col] = round(.72 + .018*(y-365)/100 + .012*(x-1200)/300,4)
        elif category == 3:  # front of mug curved slightly toward the camera
            z = .477 + .028*(y-355)/293 - .009*(1-((x-685)/120)**2)
            depth[row][col] = round(max(.45,min(.52,z)),4)
        elif category == 4:  # hand/forearm, varying across the visible limb
            z = .446 + .052*(x-740)/1180 + .009*(y-440)/200
            depth[row][col] = round(max(.43,min(.53,z)),4)
a['depth']['values_m'] = depth
a['depth']['status'] = 'estimated'

# Anatomical anchors on the last permitted RGB image, in source pixels. Exact
# locations of occluded finger joints are anatomical inferences, not visible observations.
reference = {
    0:(1080,506,.040),1:(930,464,.016),2:(850,477,.005),3:(765,487,-.001),4:(689,488,-.005),
    5:(933,438,.019),6:(871,398,.011),7:(808,367,.006),8:(756,352,0.0),
    9:(948,424,.026),10:(890,385,.021),11:(836,359,.016),12:(786,346,.012),
    13:(966,416,.032),14:(911,385,.028),15:(865,368,.025),16:(827,358,.023),
    17:(985,418,.038),18:(937,395,.035),19:(902,378,.033),20:(866,368,.032),
}
# Motion anchors were read from each of the 21 history frames in cup-001; this
# version transfers the whole hand relative to those anchors and refines the grip.
index_x = [1175,1160,1137,1120,1094,1070,1046,1020,998,971,944,919,893,868,845,821,804,789,776,765,756]
index_y = [334,335,337,335,333,334,335,335,333,332,332,332,333,335,337,339,340,342,344,347,352]
thumb_x = [1150,1136,1113,1097,1072,1050,1029,1008,990,966,944,920,887,856,829,799,770,744,720,703,689]
thumb_y = [480,479,478,477,477,478,479,480,482,485,489,492,495,493,490,488,488,490,489,489,488]
for i,state in enumerate(a['history']):
    eef_z = state['eef']['position_camera_m'][2]
    dx = index_x[i]-756
    dy = index_y[i]-352
    for j,(rx,ry,dz) in reference.items():
        x,y = rx+dx,ry+dy
        if j in (1,2,3,4):
            # Thumb bends toward the mug. Endpoints are direct visual anchors;
            # intermediate joints are proportional positions along that digit.
            fraction = {1:0.18,2:0.48,3:0.76,4:1.0}[j]
            base_x = 930+dx
            base_y = 464+dy
            x = base_x + fraction*(thumb_x[i]-base_x)
            y = base_y + fraction*(thumb_y[i]-base_y)
            if j == 4:
                x,y = thumb_x[i],thumb_y[i]
        if j in (12,16,20) and i < 12:
            # These distal fingers merge in blur. Their placement is inferred.
            y += 5
        status = 'inferred'
        if j == 4 and i >= 15:
            status = 'visible'
        if j == 8 and i >= 15:
            status = 'visible'
        if j == 0 and i >= 10:
            status = 'visible'
        z = round(eef_z+dz,4)
        state['landmarks'][j] = {'id':j,'uv':[round((x+.5)/1920,6),round((y+.5)/1080,6)],
                                 'z_m':z,'status':status}
    state['notes'] = ('Точки 0–20 оценены по контуру кисти и собственной анатомической '
                      'интерполяции; видимость указана отдельно. Перекрытые фаланги '
                      'и почти все глубины inferred, особенно на дальних пальцах.')
    # Keep centre and opening from the independently estimated first attempt;
    # update only metric camera x/y after changing the K hypothesis.
    old_eef = state['eef']
    z = old_eef['position_camera_m'][2]
    old_x = 960+old_eef['position_camera_m'][0]*1280/z
    old_y = 540+old_eef['position_camera_m'][1]*1280/z
    old_eef['position_camera_m'] = [round((old_x-cx)*z/fx,5),round((old_y-cy)*z/fy,5),z]

a['assumptions'] = [
    'Вторая попытка использует ту же историю 21 кадра и границу 6,0 с, что cup-001; будущее не просматривалось.',
    'K и плоскость стола согласованы с видимой высотой кружки при её измеренной высоте 0,11 м; реальная калибровка неизвестна.',
    'Глубина столешницы вычислена пересечением луча с оценённой плоскостью только внутри вручную размеченного полигона.',
    'Глубина кружки, наушников и руки интерполирована по видимым контурам и опорным оценкам, а не измерена сенсором.',
    'Скрытые суставы восстановлены по видимому контуру кисти и анатомической непрерывности; статус inferred не означает наблюдение.',
    'EEF — условный центр хвата; единичный quaternion оставлен соглашением, а не измеренной ориентацией кисти.',
]
a['limitations'] = [
    'Монокулярный RGB не определяет единственную метрическую геометрию: ошибка K, наклона стола и формы кружки связана между собой.',
    'Полная глубина столешницы является геометрической экстраполяцией; особенно ненадёжны дальние и крайние области.',
    'Перекрытые пальцы и скрытые стороны кисти имеют высокую неопределённость; полные 21 точки являются предположением.',
    'Полигоны приблизительны: около краёв глубина может принадлежать соседнему объекту или фону.',
    'Человеческая рука и условная EEF отличаются от обучающей кинематики Panda. Техническая упаковка не подтверждает качества прогноза.',
    'Отложенный фрагмент, 3D ADE/FDE и физическая калибровка не проверялись.',
]
q.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
