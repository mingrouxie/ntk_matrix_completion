import pandas as pd
import pdb
from utils.path_constants import ZEO_1_PRIOR

NAME = 'ZEO1'

new = {}
new['a'] = 43.3056
new['b'] = 43.3056
new['c'] = 25.0010
new['spacegroup'] =  141
new['maximum_free_sphere'] = 9.45 
new['included_sphere_diameter'] = 11.54
new['diffused_sphere_diameter_a'] = (9.35642) #+ 9.22357) / 2
new['diffused_sphere_diameter_b'] = (9.35642) # + 9.22357) / 2
new['diffused_sphere_diameter_c'] = (11.19646) # + 11.41629) / 2
new['volume'] = 46900
new['ring_size_0'] = 16
new['ring_size_1'] = 12

df = pd.DataFrame([])
series = pd.Series(data=new)
series.name = NAME
df = df.append(series)
pdb.set_trace()
df.to_pickle(ZEO_1_PRIOR)

# diffused_sphere_diameter calculated using ZEO++ http://www.zeoplusplus.org/examples.html 

# s1: refinement of calcined ZEO-1 against cRED data.
# s2: Rietveld refinement calcined ZEO-1
# s3: Rietveld refinement as made ZEO-1

# largest included sphere, the largest free sphere and the largest included sphere along free sphere path
# /home/mrx/zeo_1/analysis/s1_res.res    3.87534 1.15101  2.72992

# largest; a direction; c direction
# /home/mrx/zeo_1/analysis/s1_resex.res    3.87534 1.15101  2.72992  1.15101  1.15101  1.15101  2.71900  2.72992  2.72609  
# /home/mrx/zeo_1/analysis/s2_resex.res    11.19646 9.35642  11.19646  9.35641  9.35642  9.35642  11.19646  10.84439  11.19646
# /home/mrx/zeo_1/analysis/s3_resex.res    11.41629 9.22357  11.41629  9.22357  9.22357  9.22357  11.41629  11.41629  11.41629

