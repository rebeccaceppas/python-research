import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

birddata = pd.read_csv('./files/bird_tracking.csv')

ix = birddata.bird_name == 'Eric'
x, y = birddata.longitude[ix], birddata.latitude[ix]

plt.figure(figsize=(7,7))
plt.plot(x,y,'b.')
plt.title('Eric\'s trajectory')
plt.savefig('trajectory.jpg')

bird_names = pd.unique(birddata.bird_name)

plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ia = birddata.bird_name == bird_name
    x, y = birddata.longitude[ia], birddata.latitude[ia]
    plt.plot(x,y,'.', label=bird_name)
    
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend(loc='lower right')
plt.savefig('trajectories.jpg')

plt.figure(figsize=(7,7))
ix = birddata.bird_name == 'Eric'
speed = birddata.speed_2d[ix]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0,30,20), density=True)
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.savefig('hist.jpg')

timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
        (birddata.date_time.iloc[k][:-3], '%Y-%m-%d %H:%M:%S'))

birddata['timestamp'] = pd.Series(timestamps, index=birddata.index)

times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]
elapsed_days = []
for t in elapsed_time:
    elapsed_days.append(t / datetime.timedelta(days=1))

plt.figure(figsize=(7,7))
plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel('Observation')
plt.ylabel('Elapsed time (days)')
plt.savefig('timeplot.jpg')

next_day = 1
indices = []
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        indices.append(i)
    else:
        daily_mean_speed.append(np.mean(birddata.speed_2d[indices]))
        next_day += 1
        indices = []

plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
plt.savefig('daily mean speeds.jpg')

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for name in bird_names:
    ix = birddata['bird_name'] == name
    x,y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x,y,'.', transform=ccrs.Geodetic(), label=name)

plt.legend(loc='upper left')
plt.savefig('map.jpg')