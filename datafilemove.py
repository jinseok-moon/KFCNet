import os
from pathlib import Path
import shutil

dirlst = []
rootdir = './dataset'

# stlst = ['구이', '국', '기타', '김치', '나물', '떡', '면', '무침', '밥', '볶음', '쌈', '음청류', '장',
#  '장아찌', '적', '전', '전골', '조림', '죽', '찌개', '찜', '탕', '튀김', '해물', '회']
#
# for file in os.listdir(rootdir):
#     d = os.path.join(rootdir, file)
#     if os.path.isdir(d):
#         for dirs in stlst:
#             e = os.path.join(d, dirs)
#             if os.path.exists(e):
#                 shutil.rmtree(e)

# for path in dirlst:
#     p = Path(path).absolute()
#     parent_dir = p.parents[1]
#     p.rename(parent_dir / p.name)