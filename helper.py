import numpy as np
import pandas as pd
import scipy
import scipy.stats

import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

import statsmodels.api as sm
from statsmodels.formula.api import ols

from statsmodels.stats.anova import AnovaRM

def successive(frames):
    result = list()
    i = 0
    if len(frames) == 0:
        return list()
    else:
        while i in range(len(frames)):
            if i == 0:
                newlist = [frames[i]]
            elif frames[i] - frames[i - 1] == 1:
                newlist.append(frames[i])
            else:
                result.append(newlist)
                newlist = [frames[i]]
            i += 1
        result.append(newlist)
        return result

def representation(frames, x=1):
    assert x == 0 or x == 1 or x == 2, 'Input 0, 1, or 2 for first, middle, or last frame respectively'
    sf = successive(frames)
    result = list()
    for i, j in enumerate(sf):
        if x == 0:
            result.append(j[0])
        elif x == 1:
            result.append(round((j[0] + j[-1]) / 2))
        else:
            result.append(j[-1])
    return result

def check_backwards(indices_for_eval, target, length):
    '''
    indices_for_eval: 1D array of indices for evaluation
    target: 1D array of True, False for whole frames
    length: positive int
    '''
    def make_window(frame):
        return np.linspace(frame, frame - length + 1, num=length, endpoint=True, dtype=int)
    indices_for_eval = indices_for_eval[indices_for_eval >= length - 1].reshape(-1, 1)
    back = target[np.apply_along_axis(make_window, 1, indices_for_eval)].any(axis=1).reshape(-1)

    front = []
    for i in indices_for_eval[indices_for_eval < length -1]:
        front.append(target[np.arange(i+1, dtype=int)].any())
    front = np.array(front)

    return indices_for_eval[np.concatenate((front, back)).astype(bool)].reshape(-1)

def lininterpol(df, bodyparts, ll_crit, absolute=True):
    '''
    df: DataFrame from pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    bodyparts: list of bodyparts of interest
    ll_crit: real number in [0, 1) or list of real numbers in [0, 1)
        In the first case, the same value of ll_crit is applied to all bodyparts.
        In the second case, each value of ll_crit in the list is applied to corresponding bodyparts.
    absolute: bool or list of bools
        If absolute=True, the cutoff criterion is ll_crit itself.
        If absolute=False, the cutoff criterion is the int(# frames * ll_crit)-th lowest likelihood for each bodoypart.
        If absoulte is list, each boolean value in the list is applied to corresponding bodyparts.
    Returns:
    new_df: pd.DataFrame, bad values cut off and interpolated linearly.
    start, end: the indices in the original df corresponding to the first/last index of new_df.
        If df starts/ends with bad values, those are all cut off and not interpolated; so the first/last index of new_df may not represent the first index of df.
    '''
    numbodyparts = len(bodyparts)
    numframes = len(df)

    values = df[bodyparts].values.reshape(-1, numbodyparts, 3).transpose([1, 0, 2]) # np array of shape (numbodyparts, numframes, 3)
    
    if type(absolute) == bool:
        absolute = [absolute] * numbodyparts
    
    if type(ll_crit) == float:
        ll_crit = [ll_crit] * numbodyparts

    mins = []
    for i in range(numbodyparts):
        if absolute[i]:
            mins.append(ll_crit[i])
        else:
            cutoff_index = values[i, :, -1].argsort()[int(numframes * ll_crit[i])]
            mins.append(values[i, cutoff_index, -1])

    mins = np.array(mins).reshape(-1, 1) # np array of shape (numbodyparts, 1)
    
    good = values[:, :, -1] >= mins # np array of shape (numbodyparts, numframes), T or F

    assert good.all(axis=0).sum() >= 2, 'Likelihood criterion too high'

    start, end = np.where(good.all(axis=0))[0][0], np.where(good.all(axis=0))[0][-1]

    values = values[:, start:(end + 1), :] # np array of shape (numbodyparts, # frames for use, 3)
    good = good[:, start:(end + 1)] # np array of shape (numbodyparts, # frames for use)

    for i in range(numbodyparts):
        bad0 = np.array(representation(np.where(~good[i])[0], x=0)).reshape(-1, 1)
        bad1 = np.array(representation(np.where(~good[i])[0], x=2)).reshape(-1, 1)
        bads = np.concatenate((bad0, bad1), axis=1)

        for j in range(bads.shape[0]):
            prev_frame = int(bads[j, 0] - 1)
            next_frame = int(bads[j, 1] + 1)
            values[i, prev_frame:next_frame, :-1] = np.linspace(values[i, prev_frame, :-1], values[i, next_frame, :-1], num=(next_frame - prev_frame), endpoint=False)

    tuples = []
    for bp in bodyparts:
        tuples.append((bp, 'x'))
        tuples.append((bp, 'y'))


    new_df = pd.DataFrame(values[:, :, :-1].transpose([1, 0, 2]).reshape(-1, 2 * numbodyparts), columns=pd.MultiIndex.from_tuples(tuples))

    '''
    returns:
        (new_df of index=df.index, columns consisting of (bodypart, x) and (bodypart, y))
        start,
        end
    '''
    return new_df, start, end

def temporal_density(arr, time, FPS):
    '''
    arr: 1D array of True, False
    time: float, length of moving bin (s)
    FPS: FPS of the video
    '''
    length = int(time * FPS)
    return np.convolve(arr.astype(float), np.ones(length) / length, mode='same') * FPS

def return_bout1(filepath, params, latency1=3.0, latency2=1.0,
                 ll_crit=0.9, absolute=True, interval=0.2, FPS=24.0):
    df = pd.read_csv(filepath, index_col=0, header=[1, 2], skiprows=0)
    coords, start, end = lininterpol(df, ['Mouth', 'R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)

    trayx, trayy, dist0, dist1 = params

    coords['Condition 0R'] = np.linalg.norm(coords['Mouth'].values - coords['R_hand'].values, axis=1) < dist0
    coords['Condition 0L'] = np.linalg.norm(coords['Mouth'].values - coords['L_hand'].values, axis=1) < dist0
    coords['Condition 1R'] = np.linalg.norm(coords['R_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
    coords['Condition 1L'] = np.linalg.norm(coords['L_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1

    R_old = representation(coords[coords['Condition 0R']].index, x=0)
    R_new = []
    for frame in R_old:
        if len(R_new) == 0:
            R_new.append(frame)
        elif frame - R_new[-1] > FPS * interval:
            R_new.append(frame)
    
    L_old = representation(coords[coords['Condition 0L']].index, x=0)
    L_new = []
    for frame in L_old:
        if len(L_new) == 0:
            L_new.append(frame)
        elif frame - L_new[-1] > FPS * interval:
            L_new.append(frame)

    original_bout_R = check_backwards(np.array(R_new), coords['Condition 1R'].values, int(latency1 * FPS))
    bout_R = []
    for frame in R_new:
        if frame in original_bout_R:
            bout_R.append(frame)
        elif len(bout_R) > 0:
            if frame - bout_R[-1] < latency2 * FPS:
                bout_R.append(frame)

    original_bout_L = check_backwards(np.array(L_new), coords['Condition 1L'].values, int(latency1 * FPS))
    bout_L = []
    for frame in L_new:
        if frame in original_bout_L:
            bout_L.append(frame)
        elif len(bout_L) > 0:
            if frame - bout_L[-1] < latency2 * FPS:
                bout_L.append(frame)

    coords['Bout R'] = pd.Series(True, index=bout_R)
    coords['Bout L'] = pd.Series(True, index=bout_L)
    coords['Bout R'] = coords['Bout R'].fillna(False)
    coords['Bout L'] = coords['Bout L'].fillna(False)

    coords['Bout'] = (coords['Bout R'] | coords['Bout L'])

    return coords, start, end

def return_approach(filepath, params, ll_crit=0.9, absolute=True, interval=0.2, FPS=30):
    '''
    filepath: output of DLC; its bodyparts must contain 'R_hand', 'L_hand'
    params: (
        x-coord of tray,
        y-coord of tray,
        threshold of the distance btwn hand and tray
    )
    ll_crit, absolute: look at lininterpol function for further description
    interval: nonnegative float. the minimum time interval (s) btwn two contiguous events of the same hand entering the circle of radius params[2] centered at tray. default 0.5.
        For example, suppose the right hand entered the circle of radius params[2] centered at tray at 1.0s.
        In case of interval=0.5, we do not record this event until 1.5s, even if it appears to happen according to the DLC coordinates.
        This is because it is unlikely that the same hand enters the circle more than twice in 0.5s; this is rather because the labeled coordinate is inaccurate for a few frames.
        Hence, we exclude those events happening multiple times in some small time interval, so that our calculation is more compatible with reality.
    FPS: FPS of the video. default 30
    Returns:
    coords: DataFrame. coords['Approach'] is the Series of boolean value whether each frame is 'approach' or not.
    start, end: look at lininterpol function for further description
    '''

    df = pd.read_csv(filepath, index_col=0, header=[1, 2], skiprows=0)
    coords, start, end = lininterpol(df, ['R_hand', 'L_hand'], ll_crit=ll_crit, absolute=absolute)

    trayx, trayy, dist1 = params

    coords['R_in'] = np.linalg.norm(coords['R_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
    coords['L_in'] = np.linalg.norm(coords['L_hand'].values - np.array([[trayx, trayy]]), axis=1) < dist1
                                    
    R_old = representation(coords[coords['R_in']].index, x=0)
    R_new = []
    for frame in R_old:
        if len(R_new) == 0:
            R_new.append(frame)
        elif frame - R_new[-1] > FPS * interval:
            R_new.append(frame)

    L_old = representation(coords[coords['L_in']].index, x=0)
    L_new = []
    for frame in L_old:
        if len(L_new) == 0:
            L_new.append(frame)
        elif frame - L_new[-1] > FPS * interval:
            L_new.append(frame)

    result_indices = np.array(list(set(R_new + L_new)))
    result_indices = result_indices[result_indices.argsort()]

    coords['Approach'] = pd.Series(True, index=result_indices)
    coords['Approach'] = coords['Approach'].fillna(False)
    return coords, start, end

def return_infz(filepath, fz_x, ll_crit=0.9, absolute=True):
    '''
    filepath: output of DLC; its bodyparts must contain 'Mouth', 'R_hand', 'L_hand'
    fz_x: x-coord of food zone
    
    ll_crit, absolute: look at lininterpol function for further description
    Returns:
    coords: DataFrame. coords['In'] is the Series of boolean value whether each frame is 'In food zone' or not.
    start, end: look at lininterpol function for further description
    '''
    df = pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    coords, start, end = lininterpol(df, ['Body_center'], ll_crit=ll_crit, absolute=absolute)
    coords['In'] = (coords.values[:, 0] > fz_x)

    return coords, start, end

def get_foc_behaviors_count(df, manual, crit, filter, latency, dur):
    coords, start, end = lininterpol(df, ['Body_center'], 0.9, absolute=True)
    bcx = coords[('Body_center', 'x')].values

    head = np.full((filter - 1) // 2, fill_value=bcx[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcx[-1])
    bcx = np.concatenate((head, bcx, tail))

    bcx = np.convolve(bcx, v=np.full(filter, fill_value=1/filter), mode='valid')

    bcy = coords[('Body_center', 'y')].values

    head = np.full((filter - 1) // 2, fill_value=bcy[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcy[-1])
    bcy = np.concatenate((head, bcy, tail))

    bcy = np.convolve(bcy, v=np.full(filter, fill_value=1/filter), mode='valid')

    bc = np.vstack((bcx, bcy)).T

    v = bc[1:] - bc[:-1]
    v = np.concatenate((v[0].reshape(1, 2), (v[1:]+v[:-1])/2, v[-1].reshape(1, 2)), axis=0)
    v = np.linalg.norm(v, axis=1)

    slow = np.where(v < crit)[0]
    starts = np.array(representation(slow, x=0))
    ends = np.array(representation(slow, x=2))

    manual = manual[(manual >= start) & (manual <= end)] - start

    count = 0

    for m in manual:
        if ((ends - starts > dur) & (((starts - latency -m)*(ends-m)) <= 0)).any():
            count += 1

    return count, len(manual) - count

def get_foc_sessions(df, manual, crit, filter, latency, dur):
    coords, start, end = lininterpol(df, ['Body_center'], 0.9, absolute=True)
    bcx = coords[('Body_center', 'x')].values

    head = np.full((filter - 1) // 2, fill_value=bcx[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcx[-1])
    bcx = np.concatenate((head, bcx, tail))

    bcx = np.convolve(bcx, v=np.full(filter, fill_value=1/filter), mode='valid')

    bcy = coords[('Body_center', 'y')].values

    head = np.full((filter - 1) // 2, fill_value=bcy[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcy[-1])
    bcy = np.concatenate((head, bcy, tail))

    bcy = np.convolve(bcy, v=np.full(filter, fill_value=1/filter), mode='valid')

    bc = np.vstack((bcx, bcy)).T

    v = bc[1:] - bc[:-1]
    v = np.concatenate((v[0].reshape(1, 2), (v[1:]+v[:-1])/2, v[-1].reshape(1, 2)), axis=0)
    v = np.linalg.norm(v, axis=1)

    slow = np.where(v < crit)[0]
    starts = np.array(representation(slow, x=0))
    ends = np.array(representation(slow, x=2))

    manual = manual[(manual >= start) & (manual <= end)] - start

    result = []

    for st, en in zip(starts, ends):
        if en - st > dur and np.isin(manual, np.arange(max(0, st - latency), en + 1)).any():
            result.append([st+start, en+start])

    return np.array(result).reshape(-1, 2)

def get_lowspeed(df, crit, filter, latency, dur):
    coords, start, end = lininterpol(df, ['Body_center'], 0.9, absolute=True)
    bcx = coords[('Body_center', 'x')].values

    head = np.full((filter - 1) // 2, fill_value=bcx[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcx[-1])
    bcx = np.concatenate((head, bcx, tail))

    bcx = np.convolve(bcx, v=np.full(filter, fill_value=1/filter), mode='valid')

    bcy = coords[('Body_center', 'y')].values

    head = np.full((filter - 1) // 2, fill_value=bcy[0])
    tail = np.full(filter - 1 - len(head), fill_value=bcy[-1])
    bcy = np.concatenate((head, bcy, tail))

    bcy = np.convolve(bcy, v=np.full(filter, fill_value=1/filter), mode='valid')

    bc = np.vstack((bcx, bcy)).T

    v = bc[1:] - bc[:-1]
    v = np.concatenate((v[0].reshape(1, 2), (v[1:]+v[:-1])/2, v[-1].reshape(1, 2)), axis=0)
    v = np.linalg.norm(v, axis=1)

    slow = np.where(v < crit)[0]
    starts = np.array(representation(slow, x=0))
    ends = np.array(representation(slow, x=2))

    result = []
    for st, en in zip(starts, ends):
        if en - st > dur:
            result.append([st+start, en+start])

    return np.array(result).reshape(-1, 2)

def get_text(p):
    if np.isnan(p):
        return 'nan'
    else:
        p_around = '$p$ = {:.3f}'.format(p)
        if p >= 0.05:
            return p_around
        elif p >= 0.01:
            return '*   ({})'.format(p_around)
        elif p >= 0.001:
            return '**   ({})'.format(p_around)
        else:
            return '***   ({})'.format(p_around)

def arial_fontprop(size, weight):
    return fm.FontProperties(fname='font/arial.ttf', weight=weight, size=size)



def draw_picture(data, title, ylabel, path):
    p1 = scipy.stats.ttest_rel(data[0], data[1]).pvalue
    text1 = get_text(p1)

    ratios = data[1] / data[0]

    p2 = scipy.stats.ttest_1samp(ratios-1, popmean=0).pvalue
    text2 = get_text(p2)

    labelprop = arial_fontprop(28, 'bold')
    tickprop = arial_fontprop(25, 'bold')
    legendprop = arial_fontprop(20, 'medium')
    titleprop = arial_fontprop(30, 'bold')
    textprop = arial_fontprop(20, 'medium')
    textprop_2 = arial_fontprop(25, 'medium')


    veh_color = (0.7, 0.7, 0.7, 1.0)
    cno_color = (1.0, 0.7, 0.7, 1.0)

    names = ['Monkey A', 'Monkey B', 'Monkey C']
    markers = ['o', 's', '^']

    if (data[0] == 0).any():

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.bar([0, 1], data.mean(axis=1), color=[veh_color, cno_color], zorder=0, edgecolor='black', width=0.8)
        ax.errorbar([0, 1], data.mean(axis=1), yerr = np.vstack(([0, 0], np.std(data, axis=1, ddof=1)/np.sqrt(data.shape[1]))), color='black', capsize=10, zorder=-10, ls='none')

        for i, (name, marker) in enumerate(zip(names, markers)):
            ax.plot([0, 1], data[:, i], color='black')
            ax.scatter([0, 1], data[:, i], color='black', marker=marker, s=100, label=name)

        line = data.max() * 1.05

        ax.plot([0, 1], [line, line], color='black')
        if text1[0] != '*':
            ax.text(0.5, line, text1, fontproperties=textprop, ha='center', va='bottom')
        else:
            ax.text(0.5, line, text1, fontproperties=textprop_2, ha='center', va='bottom')

        ax.set_xlim([-1, 2])
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(['Control', 'Activation'], fontproperties=labelprop)

        ymax, ymin = line, 0

        if ymax == 0:
            return None

        scalemax, scalemin = (ymax - ymin) / 3, (ymax - ymin) / 6
        
        digit = int(np.floor(np.log10(scalemin)))
        base = 10 ** digit
        if scalemin <= base <= scalemax:
            scale = base
        elif scalemin <= 2 * base <= scalemax:
            scale = 2 * base
        elif scalemin <= 4 * base <= scalemax:
            scale = 4 * base
        elif scalemin <= 5 * base <= scalemax:
            scale = 5 * base
        elif scalemin <= 10 * base <= scalemax:
            scale = 10 * base
            digit += 1
        else:
            raise ValueError

        yticksmax = int(np.ceil(ymax / scale) + 1) * scale
        yticksmin = 0

        yticks = np.around(np.arange(yticksmin, yticksmax+scale, scale), decimals=-digit)

        # ax.set_ylim([0, line*1.25])
        ax.set_yticks(yticks)
        ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
        ax.set_ylim(yticks[[0, -1]])
        ax.set_ylabel(ylabel, fontproperties=labelprop)

        
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.4}, constrained_layout=True)
        ax = axs[0]

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.bar([0, 1], data.mean(axis=1), color=[veh_color, cno_color], zorder=0, edgecolor='black', width=0.8)
        ax.errorbar([0, 1], data.mean(axis=1), yerr = np.vstack(([0, 0], np.std(data, axis=1, ddof=1)/np.sqrt(data.shape[1]))), color='black', capsize=10, zorder=-10, ls='none')

        for i, (name, marker) in enumerate(zip(names, markers)):
            ax.plot([0, 1], data[:, i], color='black')
            ax.scatter([0, 1], data[:, i], color='black', marker=marker, s=100, label=name)

        line = data.max() * 1.05

        ax.plot([0, 1], [line, line], color='black')
        if text1[0] != '*':
            ax.text(0.5, line, text1, fontproperties=textprop, ha='center', va='bottom')
        else:
            ax.text(0.5, line, text1, fontproperties=textprop_2, ha='center', va='bottom')
        

        ax.set_xlim([-1, 2])
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(['Control', 'Activation'], fontproperties=labelprop, rotation=45)

        ymax, ymin = line, 0

        if ymax == 0:
            return None

        scalemax, scalemin = (ymax - ymin) / 3, (ymax - ymin) / 6
        
        digit = int(np.floor(np.log10(scalemin)))
        base = 10 ** digit
        if scalemin <= base <= scalemax:
            scale = base
        elif scalemin <= 2 * base <= scalemax:
            scale = 2 * base
        elif scalemin <= 4 * base <= scalemax:
            scale = 4 * base
        elif scalemin <= 5 * base <= scalemax:
            scale = 5 * base
        elif scalemin <= 10 * base <= scalemax:
            scale = 10 * base
            digit += 1
        else:
            raise ValueError

        yticksmax = int(np.ceil(ymax / scale)) * scale
        yticksmin = 0

        yticks = np.around(np.arange(yticksmin, yticksmax+scale, scale), decimals=-digit)

        # ax.set_ylim([0, line*1.25])
        ax.set_yticks(yticks)
        ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
        ax.set_ylim(yticks[[0, -1]])
        ax.set_ylabel(ylabel, fontproperties=labelprop)

        
        ax = axs[1]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.axhline(0, color='black', ls=':')

        yerr = np.std(ratios, ddof=1)/np.sqrt(len(ratios))

        ax.errorbar(1, ratios.mean()-1, yerr = yerr, color='black', capsize=10, zorder=0)

        ax.boxplot(ratios-1, meanline=True, showmeans=True, showfliers=False, widths=0.8,
                medianprops={'linewidth': 0}, meanprops={'color': 'red', 'linestyle': '-'}, boxprops={'facecolor': cno_color}, patch_artist=True, positions=[1], zorder=1,
                whis=0)

        for ratio, marker in zip(ratios, markers):
            ax.plot([0, 1], [0, ratio-1], color='black', marker=marker, markersize=10, zorder=2)

        ax.plot([1.5, 1.5], [0.0, ratios.mean()-1], color='black')
        if text2[0] != '*':
            ax.text(1.5, 0.5*(ratios.mean()-1), text2, fontproperties=textprop, ha='left', va='center', rotation=-90)
        else:
            ax.text(1.5, 0.5*(ratios.mean()-1), text2, fontproperties=textprop_2, ha='left', va='center', rotation=-90)
        

        ax.set_xlim([-1, 2])
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(['Control', 'Activation'], fontproperties=labelprop, rotation=45)

        
        ymax, ymin = max(0.0, ratios.max()-1, ratios.mean()-1+yerr), min(0.0, ratios.min()-1, ratios.mean()-1-yerr)

        scalemax, scalemin = (ymax - ymin) / 3, (ymax - ymin) / 6
        
        digit = int(np.floor(np.log10(scalemin)))
        base = 10 ** digit
        if scalemin <= base <= scalemax:
            scale = base
        elif scalemin <= 2 * base <= scalemax:
            scale = 2 * base
        elif scalemin <= 4 * base <= scalemax:
            scale = 4 * base
        elif scalemin <= 5 * base <= scalemax:
            scale = 5 * base
        elif scalemin <= 10 * base <= scalemax:
            scale = 10 * base
        else:
            raise ValueError

        yticksmax = int(np.ceil(ymax / scale)+1) * scale
        yticksmin = int(np.floor(ymin / scale) - 1) * scale

        yticks = np.around(np.arange(yticksmin, yticksmax+scale, scale), decimals=-digit)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
        ax.set_ylim(yticks[[0, -1]])

        ax.set_ylabel('Normalized Value', fontproperties=labelprop)    

    fig.tight_layout()
    fig.savefig(path)

def draw_proportion(data, title, path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [1, 10], 'hspace': 0.1})
    p = scipy.stats.ttest_rel(data[0], data[1]).pvalue
    text = get_text(p)

    labelprop = arial_fontprop(30, 'bold')
    tickprop = arial_fontprop(25, 'bold')
    legendprop = arial_fontprop(20, 'medium')
    titleprop = arial_fontprop(30, 'bold')
    textprop = arial_fontprop(20, 'medium')
    textprop_2 = arial_fontprop(25, 'medium')


    veh_color = (0.7, 0.7, 0.7, 1.0)
    cno_color = (1.0, 0.7, 0.7, 1.0)

    ax = axs[0]
    ax.set_ylim([0, 1])
    ax.set_xlim([-1, 2])
    ax.plot([0, 1], [0, 0], color='black', clip_on=False)

    if text[0] != '*':
        ax.text(0.5, 0, text, fontproperties=textprop, ha='center', va='bottom')
    else:
        ax.text(0.5, 0, text, fontproperties=textprop_2, ha='center', va='bottom')

    ax.set_axis_off()

    names = ['Monkey A', 'Monkey B', 'Monkey C']
    markers = ['o', 's', '^']

    ax = axs[1]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.bar([0, 1], data.mean(axis=1), color=[veh_color, cno_color], zorder=0, edgecolor='black', width=0.8)

    for i, (name, marker) in enumerate(zip(names, markers)):
        ax.plot([0, 1], data[:, i], color='black')
        ax.scatter([0, 1], data[:, i], color='black', marker=marker, s=100, label=name)

    ax.set_xlim([-1, 2])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(['Control', 'Activation'], fontproperties=labelprop)
    ax.tick_params(axis='x', pad=5)

    yticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # ax.set_ylim([0, line*1.25])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ax.get_yticks(), fontproperties=tickprop)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Proportion', fontproperties=labelprop)

    
    fig.savefig(path)