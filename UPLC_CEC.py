
#########################################

#import sys
#sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.8/site-packages'))
import os
import glob

#load required pacakges (as always)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import json

#packages for functions imported from hplc.py
import scipy.signal
import scipy.optimize
import scipy.special
import tqdm
import seaborn as sns
import matplotlib

#Functions and object class imported from cremerlab hplc.py#######################################
class Chromatogram(object):
    """
    Base class for dealing with HPLC chromatograms
    """

    def __init__(self, file=None, time_window=None, baselinecorrection = True,
                 cols={'time': 'time_min', 'intensity': 'intensity_mV'},
                 csv_comment='#'):
        """
        Instantiates a chromatogram object on which peak detection and quantification
        is performed.
        Parameters
        ----------
        file: str or pandas DataFrame, optional
            The path to the csv file of the chromatogram to analyze or
            the pandas DataFrame of the chromatogram. If None, a pandas DataFrame
            of the chromatogram must be passed.
        dataframe : pandas DataFrame, optional
            a Pandas Dataframe of the chromatogram to analyze. If None,
            a path to the csv file must be passed
        time_window: list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        cols: dict, keys of 'time', and 'intensity', optional
            A dictionary of the retention time and intensity measurements
            of the chromatogram. Default is
            `{'time':'time_min', 'intensity':'intensity_mV'}`.
        csv_comment: str, optional
            Comment delimiter in the csv file if chromatogram is being read
            from disk.
        """

        # Peform type checks and throw exceptions where necessary.
        if file is None:
            raise RuntimeError(f'File path or dataframe must be provided')
        if (type(file) is not str) & (type(file) is not pd.core.frame.DataFrame):
            raise RuntimeError(
                f'Argument must be either a filepath or pandas DataFrame. Argument is of type {type(file)}')
        if (time_window is not None):
            if type(time_window) != list:
                raise TypeError(f'`time_window` must be of type `list`. Type {type(time_window)} was proivided')
            if len(time_window) != 2:
                raise ValueError(
                    f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')

        # Assign class variables
        self.time_col = cols['time']
        self.int_col = cols['intensity']

        # Load the chromatogram and necessary components to self.
        if type(file) is str:
            dataframe = pd.read_csv(file, comment='#')
        else:
            dataframe = file
        self.df = dataframe

        # Prune to time window
        if time_window is not None:
            self.crop(time_window)
        else:
            self.df = dataframe

        # Correct for a negative baseline
        df = self.df
        first_int = df[self.int_col].iloc[0]
        if baselinecorrection:
            intensity = df[self.int_col] - first_int

            for intns in range(0,len(intensity)):
                 df.iloc[intns, 2] = intensity[intns]

        # Blank out vars that are used elsewhere
        self.window_df = None
        self.window_props = None
        self.peaks = None
        self.peak_df = None

    def crop(self, time_window=None, return_df=False):
        """
        Restricts the time dimension of the DataFrame
        Parameters
        ----------
        time_window : list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        return_df : bool
            If `True`, the cropped DataFrame is
        Returns
        -------
        cropped_df : pandas DataFrame
            If `return_df = True`, then the cropped dataframe is returned.
        """
        if type(time_window) != list:
            raise TypeError(f'`time_window` must be of type `list`. Type {type(time_window)} was proivided')
        if len(time_window) != 2:
            raise ValueError(
                f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')
        self.df = self.df[(self.df[self.time_col] >= time_window[0]) &
                          (self.df[self.time_col] <= time_window[1])]
        if return_df:
            return self.df

    def backgroundsubstraction(self, num_iterations=10, return_df=False):
        """
        Substracts background for entire chromatogram using the algorithm by Miroslav Morhac et al

        Parameters
        ----------
        num_iterations : int
            The number of iterations to run. For each iteration one additional pixel is included and one should chose the number of iterations as the typical width of a peak
        return_df : bool
            If `True`, then chromatograms (before and after background correction) are returned
        Returns
        -------
        corrected_df : pandas DataFrame
            If `return_df = True`, then the original and the corrected chromatogram are returned.
        """
        df = self.df
        try:
            intensity = self.df[self.int_col + "_nobackgroundcorrection"].values
        except:
            intensity = self.df[self.int_col].values

        intensity_old = intensity.copy()
        intensity = intensity * np.heaviside(intensity, 0)
        # transform to log scale
        intensity_transf = np.log(np.log(np.sqrt(intensity + 1) + 1) + 1)
        # start itteration
        for il in range(0, num_iterations):
            intensity_transf_new = intensity_transf.copy()
            for i in range(il, intensity_transf.shape[0] - il):
                intensity_transf_new[i] = min(intensity_transf[i],
                                              0.5 * (intensity_transf[i + il] + intensity_transf[i - il]))
            intensity_transf = intensity_transf_new
        # transform back
        intensity = np.power(np.exp(np.exp(intensity_transf) - 1.) - 1., 2.) - 1.
        self.df[self.int_col] = intensity_old - intensity
        self.df[self.int_col + '_nobackgroundcorrection'] = intensity_old
        self.df[self.int_col + '_background'] = intensity

        if return_df:
            return self.df

    def _assign_peak_windows(self, prominence, rel_height, buffer, manual_peak_positions=None,
                             baselinecorrection=False):
        """
        Breaks the provided chromatogram down to windows of likely peaks.
        Parameters
        ----------
        prominence : float,  [0, 1]
            The promimence threshold for identifying peaks. Prominence is the
            relative height of the normalized signal relative to the local
            background. Default is 1%.
        rel_height : float, [0, 1]
            The relative height of the peak where the baseline is determined.
            Default is 95%.
        buffer : positive int
            The padding of peak windows in units of number of time steps. Default
            is 100 points on each side of the identified peak window.
        manual_peak_positions : list of floats
            to provide peak position by hand instead of estimating peak positions via script
        Returns
        -------
        windows : pandas DataFrame
            A Pandas DataFrame with each measurement assigned to an identified
            peak or overlapping peak set. This returns a copy of the chromatogram
            DataFrame with  a column  for the local baseline and one column for
            the window IDs. Window ID of -1 corresponds to area not assigned to
            any peaks
        """
        for param, param_name, param_type in zip([prominence, rel_height, buffer],
                                                 ['prominence', 'rel_height', 'buffer'],
                                                 [float, float, int]):
            if type(param) is not param_type:
                raise TypeError(
                    f'Parameter {param_name} must be of type `{param_type}`. Type `{type(param)}` was supplied.')
        if (prominence < 0) | (prominence > 1):
            raise ValueError(f'Parameter `prominence` must be [0, 1].')
        if (rel_height < 0) | (rel_height > 1):
            raise ValueError(f'Parameter `rel_height` must be [0, 1].')
        if (buffer < 0):
            raise ValueError('Parameter `buffer` cannot be less than 0.')

        # Correct for a negative baseline
        df = self.df
        intensity = self.df[self.int_col].values
        norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min())

        # Identify the peaks and get the widths and baselines
        if manual_peak_positions == None:
            peaks, _ = scipy.signal.find_peaks(norm_int, prominence=prominence)
        else:
            timew = self.df[self.time_col].values
            deltatw = (timew[-1] - timew[0]) / np.float64(timew.shape[0])
            peaks = np.int_((np.array(manual_peak_positions) - timew[
                0]) / deltatw)  # peeak position in descrete step of time-window
            # print("peaks detected at times : "+str(self.df[self.time_col].values[peaks]))
            #
            # print(timew[peaks]) #time of detected peaks
        if len(peaks) > 0 and peaks[
            0] == 0:  # do not use first peak if it is exactly at left boundrary #maybe better solution
            peaks = peaks[1:]
        self.peaks_inds = peaks

        # need to fix, not needed for manual peak position
        out = scipy.signal.peak_widths(intensity, peaks,
                                       rel_height=rel_height)
        _, heights, left, right = out
        widths, _, _, _ = scipy.signal.peak_widths(intensity, peaks,
                                                   rel_height=0.5)

        ###
        # Set up the ranges
        ranges = []
        for l, r in zip(left, right):
            if (l - buffer) < 0:
                l = 0
            else:
                l -= buffer
            if (r + buffer) > len(norm_int):
                r = len(norm_int)
            else:
                r += buffer
            ranges.append(np.arange(np.round(l), np.round(r), 1))

        # Identiy subset ranges and remove
        valid = [True] * len(ranges)
        for i, r1 in enumerate(ranges):
            for j, r2 in enumerate(ranges):
                if i != j:
                    if set(r2).issubset(r1):
                        valid[j] = False

        # Keep only valid ranges and baselines
        ranges = [r for i, r in enumerate(ranges) if valid[i] is True]
        baselines = [h for i, h in enumerate(heights) if valid[i] is True]

        # Copy the dataframe and return the windows

        window_df = df.copy(deep=True)
        window_df.sort_values(by=self.time_col, inplace=True)
        window_df['time_idx'] = np.arange(len(window_df))
        for i, r in enumerate(ranges):
            window_df.loc[window_df['time_idx'].isin(r),
            'window_idx'] = int(i + 1)
            window_df.loc[window_df['time_idx'].isin(r),
            'baseline'] = baselines[i]
        window_df.dropna(inplace=True)

        # Convert this to a dictionary for easy parsing
        window_dict = {}

        time_step = np.mean(np.diff(self.df[self.time_col].values))
        for g, d in window_df.groupby('window_idx'):
            _peaks = [p for p in peaks if (p in d['time_idx'].values) and (d[d['time_idx'] == p][self.int_col].values[
                                                                               0] > 0)]  # ignores peaks where intensity is smaller zero
            peak_inds = [x for _p in _peaks for x in np.where(peaks == _p)[0]]
            if baselinecorrection:
                _dict = {'time_range': d[self.time_col].values,
                         'intensity': d[self.int_col] - baselines[i],  # ? is this the good correction to make
                         'intensity_nobaselinecorrection': d[self.int_col],  # added
                         'num_peaks': len(_peaks),
                         'amplitude': [d[d['time_idx'] == p][self.int_col].values[0] - baselines[i] for p in _peaks],
                         'amplitude_nobaselinecorrection': [d[d['time_idx'] == p][self.int_col].values[0] for p in
                                                            _peaks],
                         'location': [d[d['time_idx'] == p][self.time_col].values[0] for p in _peaks],
                         'width': [widths[ind] * time_step for ind in peak_inds]
                         }
            else:
                _dict = {'time_range': d[self.time_col].values,
                         'intensity': d[self.int_col],  # added
                         'num_peaks': len(_peaks),
                         'amplitude': [d[d['time_idx'] == p][self.int_col].values[0] for p in _peaks],
                         'location': [d[d['time_idx'] == p][self.time_col].values[0] for p in _peaks],
                         'width': [widths[ind] * time_step for ind in peak_inds]
                         }
            window_dict[g] = _dict
        self.window_props = window_dict
        return window_df

    def _compute_skewnorm(self, x, *params):
        R"""
        Computes the lineshape of a skew-normal distribution given the shape,
        location, and scale parameters
        Parameters
        ----------
        x : float or numpy array
            The time dimension of the skewnorm
        params : list, [amplitude, loc, scale, alpha]
            Parameters for the shape and scale parameters of the skewnorm
            distribution.
                amplitude : float; > 0
                    Height of the peak.
                loc : float; > 0
                    The location parameter of the distribution.
                scale : float; > 0
                    The scale parameter of the distribution.
                alpha : float; >
                    THe skew shape parater of the distribution.
        Returns
        -------
        scaled_pdf : float or numpy array, same shape as `x`
            The PDF of the skew-normal distribution scaled with the supplied
            amplitude.
        Notes
        -----
        This function infers the parameters defining skew-norma distributions
        for each peak in the chromatogram. The fitted distribution has the form

        .. math::
            I = 2I_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]
        where :math:`I_\text{max}` is the maximum intensity of the peak,
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.
        """
        amp, loc, scale, alpha = params
        _x = alpha * (x - loc) / scale
        normfactor = 1  # np.sqrt(2 * np.pi * scale**2)**-1 *
        norm = normfactor * np.exp(-(x - loc) ** 2 / (2 * scale ** 2))
        cdf = (1 + scipy.special.erf(_x / np.sqrt(2)))
        return amp * norm * cdf

    def _fit_skewnorms(self, x, *params):
        R"""
        Estimates the parameters of the distributions which consititute the
        peaks in the chromatogram.
        Parameters
        ----------
        x : float
            The time dimension of the skewnorm
        params : list of length 4 x number of peaks, [amplitude, loc, scale, alpha]
            Parameters for the shape and scale parameters of the skewnorm
            distribution. Must be provided in following order, repeating
            for each distribution.
                amplitude : float; > 0
                    Height of the peak.
                loc : float; > 0
                    The location parameter of the distribution.
                scale : float; > 0
                    The scale parameter of the distribution.
                alpha : float; >
                    THe skew shape parater of the distribution.
        Returns
        -------
        out : float
            The evaluated distribution at the given time x. This is the summed
            value for all distributions modeled to construct the peak in the
            chromatogram.
        """
        # Get the number of peaks and reshape for easy indexing
        n_peaks = int(len(params) / 4)
        params = np.reshape(params, (n_peaks, 4))
        out = 0

        # Evaluate each distribution
        for i in range(n_peaks):
            out += self._compute_skewnorm(x, *params[i])
        return out

    def _estimate_peak_params(self, boundpars=None, verbose=True, baselinecorretion=False):
        R"""
        For each peak window, estimate the parameters of skew-normal distributions
        which makeup the peak(s) in the window.
        Parameters
        ----------
        boundpars : list
            Used to provide boundaries for peak fittng function
            8 boundaries provided (default value in parenthesis)
                1. lower boundary amplitude (0)
                2. lower boundary time window; difference to peak position
                3. lower boundary peak width (0.0)
                4. lower boundary skew parameter (-np.inf)
                5. upper boundary amplitude (np.inf)
                6. upper boundary time window; difference to peak position
                7. upper boundary peak width (np.inf)
                8. upper boundary skew parameter (np.inf)


        verbose : bool
            If `True`, a progress bar will be printed during the inference.
        """
        if self.window_props is None:
            raise RuntimeError('Function `_assign_peak_windows` must be run first. Go do that.')
        if verbose:
            iterator = tqdm.tqdm(self.window_props.items(), desc='Fitting peak windows...')
        else:
            iterator = self.window_props.items()
        peak_props = {}
        for k, v in iterator:
            window_dict = {}
            # Set up the initial guess
            p0 = []
            bounds = [[], []]
            for i in range(v['num_peaks']):

                # if v['amplitude'][i]<0: #ignore peaks with negative amplitude
                #    print("peak "+str(i)+"amplitude: "+str(v['amplitude'][i])+" at position: "+str(v['location'][i]))
                #    print("Warning: negative amplitude. Value before reset: "+str(v['amplitude'][i])+" at time "+str(v['location'][i]))
                #    v['amplitude'][i]=0
                # if v['amplitude_nobaselinecorrection'][i]<0: #ignore peaks with negative amplitude
                #    print("peak "+str(i)+"amplitude: "+str(v['amplitude'][i])+" at position: "+str(v['location'][i]))
                #    print("Warning: negative amplitude. Value before reset: "+str(v['amplitude_nobaselinecorrection'][i])+" at time "+str(v['location'][i]))
                #    v['amplitude_nobaselinecorrection'][i]=0
                p0.append(v['amplitude'][i])  # before w/ baseline correction

                p0.append(v['location'][i]),
                p0.append(max(v['width'][i] / 4., 0.05))  # scale parameter
                p0.append(0)  # Skew parameter, starts with assuming Gaussian

                # REMOVE OPTION NOBASELINE..
                # Set boundaries of fitting
                if boundpars == None:  # standard values
                    # lower boundaries
                    bounds[0].append(v['amplitude'][i] * 0.5)  # min amplitude
                    bounds[0].append(v['time_range'].min())  # min peak position
                    bounds[0].append(0)  # min width
                    bounds[0].append(-np.inf)  # min skew parameter
                    # upper boundaries
                    bounds[1].append(v['amplitude'][i] * 2 + 2)  # max amplitude #?before with baselinecorrection
                    bounds[1].append(v['time_range'].max())  # max peak position
                    bounds[1].append(np.inf)  # max width
                    bounds[1].append(np.inf)  # skew parameter
                else:
                    # lower boundaries
                    bounds[0].append(v['amplitude'][i] * boundpars[0])
                    bounds[0].append(v['location'][i] - boundpars[1])
                    bounds[0].append(boundpars[2])
                    bounds[0].append(boundpars[3])
                    # upper boundaries
                    bounds[1].append(v['amplitude'][i] * boundpars[4] + 2)
                    bounds[1].append(v['location'][i] + boundpars[5])
                    bounds[1].append(boundpars[6])
                    bounds[1].append(boundpars[7])

                    # Perform the inference
            if len(p0) > 0:
                try:
                    print("p0")
                    print(p0)
                    print("bounds")
                    print(bounds)
                    print(np.array(p0) - np.array(bounds[0]))
                    print(np.array(bounds[1]) - np.array(p0))
                    popt, _ = scipy.optimize.curve_fit(self._fit_skewnorms, v['time_range'],
                                                       v['intensity'], p0=p0, bounds=bounds,
                                                       maxfev=int(1E6))
                    # Assemble the dictionary of output
                    if v['num_peaks'] > 1:
                        popt = np.reshape(popt, (v['num_peaks'], 4))
                    else:
                        popt = [popt]
                    for i, p in enumerate(popt):
                        window_dict[f'peak_{i + 1}'] = {
                            'retention_time_firstguess': v['location'][i],
                            'amplitude': p[0],
                            'retention_time': p[1],
                            'std_dev': p[2],
                            'alpha': p[3],
                            'area': self._compute_skewnorm(v['time_range'], *p).sum() * float(
                                v['time_range'][-1] - [v['time_range'][0]]) / float(v['time_range'].shape[0] - 1)}

                    peak_props[k] = window_dict
                except RuntimeError:
                    print(
                        'Warning: Parameters could not be inferred for one peak')  # ? or there is no peak in that window
                    print(p0)
                    print(v['time_range'].max())
                    print(v['time_range'].min())
                    print(v['intensity'])
            else:
                pass
                # print("Warning: Window without any peak to search for")
        self.peak_props = peak_props
        return peak_props

    def quantify(self, time_window=None, prominence=1E-3, rel_height=1.0,
                 buffer=100, manual_peak_positions=None, boundpars=None, peakpositionsonly=False, verbose=True):
        R"""
        Quantifies peaks present in the chromatogram
        Parameters
        ----------
        time_window: list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        prominence : float,  [0, 1]
            The promimence threshold for identifying peaks. Prominence is the
            relative height of the normalized signal relative to the local
            background. Default is 1%.
        rel_height : float, [0, 1]
            The relative height of the peak where the baseline is determined.
            Default is 95%.
        buffer : positive int
            The padding of peak windows in units of number of time steps. Default
            is 100 points on each side of the identified peak window.
        manual_peak_positions : list
            Manual peak positions of chromatogram. When not provided, autodetection of peaks
        peakpositonsonly : bool
            If ture, only the peak positions will be provided and no full analys of peaks is included
        verbose : bool
            If True, a progress bar will be printed during the inference.
        Returns
        -------
        peak_df : pandas DataFrame
            A dataframe containing information for each detected peak. If peakpositononly=True, this is just a list of peak positions
        Notes
        -----
        This function infers the parameters defining skew-norma distributions
        for each peak in the chromatogram. The fitted distribution has the form

        .. math::
            I = 2I_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]
        where :math:`I_\text{max}` is the maximum intensity of the peak,
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.
        """

        if time_window is not None:
            dataframe = self.df
            self.df = dataframe[(dataframe[self.time_col] >= time_window[0]) &
                                (dataframe[self.time_col] <= time_window[1])].copy(deep=True)

            # Assign the window bounds (contains peak autodetection)
        _ = self._assign_peak_windows(prominence, rel_height, buffer, manual_peak_positions=manual_peak_positions)

        # stop script here if only peak positions are wanted
        if peakpositionsonly:
            peakpositions = []
            # print(self.window_props)
            iterator = self.window_props.items()
            peak_props = {}
            for k, v in iterator:  # go through every window
                window_dict = {}
                for i in range(v['num_peaks']):
                    peakpositions.append(v['location'][i])
            return peakpositions

        # Infer the distributions for the peaks
        peak_props = self._estimate_peak_params(boundpars=boundpars, verbose=verbose)

        # Set up a dataframe of the peak properties
        peak_df = pd.DataFrame([])
        iter = 0
        for _, peaks in peak_props.items():
            for _, params in peaks.items():
                _dict = {'retention_time': params['retention_time'],
                         'retention_time_firstguess': params['retention_time_firstguess'],
                         'scale': params['std_dev'],
                         'skew': params['alpha'],
                         'amplitude': params['amplitude'],
                         'area': params['area'],
                         'peak_idx': iter + 1}
                iter += 1
                print("type of peak_df:", type(peak_df))
                print("content of peak_df", peak_df)
                peak_df = pd.concat([peak_df, pd.DataFrame([_dict])], ignore_index=True)
                peak_df['peak_idx'] = peak_df['peak_idx'].astype(int)
        self.peak_df = peak_df

        # Compute the mixture
        time = self.df[self.time_col].values
        out = np.zeros((len(time), len(peak_df)))
        iter = 0
        for _k, _v in self.peak_props.items():
            for k, v in _v.items():
                params = [v['amplitude'], v['retention_time'],
                          v['std_dev'], v['alpha']]
                # print(params)
                # print(time)
                # print(self._compute_skewnorm(time, *params))

                out[:, iter] = self._compute_skewnorm(time, *params)
                iter += 1
        self.mix_array = out
        return peak_df

    def show(self):
        """
        Displays the chromatogram with mapped peaks if available.
        """
        sns.set()

        # Set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.int_col)

        # Plot the raw chromatogram
        ax.plot(self.df[self.time_col], self.df[self.int_col], 'k-', lw=2,
                label='raw chromatogram')

        # Compute the skewnorm mix
        if self.peak_df is not None:
            time = self.df[self.time_col].values
            # Plot the mix
            convolved = np.sum(self.mix_array, axis=1)
            ax.plot(time, convolved, 'r--', label='inferred mixture')
            for i in range(len(self.peak_df)):
                ax.fill_between(time, self.mix_array[:, i], label=f'peak {i + 1}',
                                alpha=0.5)
        ax.legend(bbox_to_anchor=(1, 1))
        fig.patch.set_facecolor((0, 0, 0, 0))
        return [fig, ax]


def batch_process(file_paths, time_window=None,  show_viz=False,
                  cols={'time' :'time_min', 'intensity' :'intensity_mV'},
                  lower_limit=None, upper_limit=None,plot_comp = False, manual_peak_positions=None ,backgroundsubstraction=True
                  ,backgroundsubstraction_iterations=80 , plot_characteristictimes=False ,plot_output=False,plot_show = False,
                  **kwargs):
    """
    Performs complete quantification of a set of HPLC data. Data must first
    be converted to a tidy long-form CSV file by using `cremerlab.hplc.convert`
    Parameters
    ----------
    file_paths : list of str
        A list of the file paths.
    time_window : list, optional
        The upper and lower and upper time bounds to consider for the analysis.
        Default is `None`, meaning the whole chromatogram is considered.
    show_viz : bool
        If `True`, the plot is printed to the screen. If `False`, a plot is
        generated and returned, just not shown.
    cols: dict, keys of 'time', and 'intensity', optional
        A dictionary of the retention time and intensity measurements of the
        chromatogram. Default is `{'time':'time_min', 'intensity':'intensity_mV'}`.
    manual_peak_positions : list
        Manual position of peaks (when not provided, autodetection of peak positions)
        Alternative a list of a list of peaks can be provided - used for the different samples processed during the batch run
    upper_limit: float
        for plotting; upper limit of y-axis, auto-scaling when not provided
    lower_limit: float
        for plotting; lower limit of y-axis, auto-scaling when not provided. upper_limit must be defined when used.
    backgroundsubstraction : bool
        Substract background signal following Morhac & Matousek (2008). To show comparison of original and corrected chromatogram use
        batch_plot function.
    backgroundsubstraction_iterations : int
        Number of neighboring time-points to use for background substraction
    plot_characteristictimes : [list_name_charact, list_time_range_characteristics]
    kwargs: dict, **kwargs
        **kwargs for the peak quantification function `cremerlab.hplc.Chromatogram.quantify`
    Returns
    --------
    chrom_df : pandas DataFrame
        A pandas DataFrame  of all of the chromatograms, indexed by file name
    peak_df : pandas DataFrame
        A pandas DataFrame of all identified peaks, indexed by file name
    fig : matplotlib.figure.Figure
        Matplotlib figure object for the chromatograms
    ax :  matplotlib AxesSubplot
        The axes of the matplotlib figure
    """

    # Instantiate storage lists
    chrom_dfs, peak_dfs, mixes = [], [], []

    # Perform the processing for each file
    for i, f in enumerate(tqdm.tqdm(file_paths, desc='Processing files...')):
        # Generate the sample id
        if '/' in f:
            file_name = f.split('/')[-1]
        else:
            file_name = f

        # Check for common file name extension
        for pat in ['.csv', '.txt']:
            if pat in file_name:
                file_name = file_name.split(pat)[0]
                continue

        # Parse the chromatogram and quantify the peaks
        chrom = Chromatogram(f, cols=cols, time_window=time_window)

        if backgroundsubstraction:
            chrom.backgroundsubstraction(num_iterations=backgroundsubstraction_iterations)

        if manual_peak_positions == None or type(manual_peak_positions[0]) != list:
            peaks = chrom.quantify(verbose=False, manual_peak_positions=manual_peak_positions, **kwargs)
        else:
            peaks = chrom.quantify(verbose=False, manual_peak_positions=manual_peak_positions[i], **kwargs)
        # Set up the dataframes for chromatograms and peaks
        _df = chrom.df
        _df['sample'] = file_name
        peaks['sample'] = file_name
        peak_dfs.append(peaks)

        print(peaks)                         ## display specific to IPython
        chrom_dfs.append(chrom.df)
        mixes.append(chrom.mix_array)

    # Concatenate the dataframe
    chrom_df = pd.concat(chrom_dfs, sort=False)
    peak_df = pd.concat(peak_dfs, sort=False)

    # Determine the size of the figure
    num_traces = len(chrom_df['sample'].unique())+1
    num_cols = int(3)
    num_rows = int(np.ceil(num_traces / num_cols))
    unused_axes = (num_cols * num_rows) - num_traces

    # Instantiate the figure
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12 * num_cols, 8 * num_rows))

    ax = ax.ravel()
    for a in ax:
        a.xaxis.set_tick_params(labelsize=15)
        a.yaxis.set_tick_params(labelsize=15)
        a.set_ylabel(cols['intensity'], fontsize=20)
        a.set_xlabel(cols['time'], fontsize=20)
    for i in range(unused_axes):
        ax[-(i + 1)].axis('off')

    # Assign samples to axes.
    mapper = {g: i for i, g in enumerate(chrom_df['sample'].unique())}
    mapper['all_chromatograms'] = num_traces - 1

    colorlist =['#0072B2', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#000000', '#D55E00', '#CC79A7', '#999999', '#B2DF8A',
                '#2C92DF', '#9E0032', '#92D050', '#6D9F00', '#808080', '#E7D4E8', '#7CB8DD', '#F4A460', '#FFEFD5', '#D2691E',
                '#7FDBFF', '#D68E23', '#B3E2CD', '#68228B', '#E1BEE7', '#90EE90', '#F5F5F5', '#FFFFFF', '#9468E0', '#1EFB90',
                '#0072B2', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#000000', '#D55E00', '#CC79A7', '#999999', '#B2DF8A',
                '#2C92DF', '#9E0032', '#92D050', '#6D9F00', '#808080', '#E7D4E8', '#7CB8DD', '#F4A460', '#FFEFD5', '#D2691E',]
    col_idx = 0

    # Plot the chromatogram
    for g, d in chrom_df.groupby('sample'):
        if backgroundsubstraction:
            ax[mapper[g]].plot(d[cols['time']], d[cols['intensity']], 'b-', lw=1.5,
                               label='after BG correct.')
            ax[mapper[g]].plot(d[cols['time']], d[cols['intensity' ] +'_nobackgroundcorrection'], 'y--', lw=1.5,
                               label='original')
            ax[mapper[g]].plot(d[cols['time']], d[cols['intensity' ] +"_background"], color='m' ,ls=':', lw=1.5,
                               label='background')
            ax[mapper['all_chromatograms']].plot(d[cols['time']], d[cols['intensity'] + '_nobackgroundcorrection'],
                                                 color=colorlist[col_idx], lw=1.5,
                                                 label=g)
        else:
            ax[mapper[g]].plot(d[cols['time']], d[cols['intensity']], 'b-', lw=1.5,
                               label='original')
            ax[mapper['all_chromatograms']].plot(d[cols['time']], d[cols['intensity']], color=colorlist[col_idx],
                                                 lw=1.5,
                                                 label=g)
        ax[mapper[g]].set_title(' '.join(g.split('_')), fontsize=12)
        col_idx = col_idx + 1

    # Plot the mapped peaks
    curgroupby =peak_df.groupby(['sample'])
    for g, d in peak_df.groupby(['sample']):
        mix = mixes[mapper[g]]
        # display(g)
        # display(d)
        # display(mix)

        convolved = np.sum(mix, axis=1)
        for i in range(len(d)):
            _m = np.array(mix[:, i])
            colorlist = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                         '#cab2d6', '#6a3d9a']
            colorc = colorlist[i % 10]
            time = np.linspace(time_window[0], time_window[1], len(_m))
            ax[mapper[g]].fill_between(time, 0, _m, alpha=0.5, label=f'peak {i + 1}', color=colorc)
            try:
                ax[mapper[g]].axvline(d.at[i, 'retention_time'], ls='--', alpha=1,
                                      color=colorc)  # print retention time of fit
                # ax[mapper[g]].axhline(d.at[i,'amplitude'],ls=':',alpha=0.5,color=colorc) #print retention time of first peak est.

            except:
                pass
        time = np.linspace(time_window[0], time_window[1], len(convolved))
        ax[mapper[g]].plot(time, convolved, '--', color='red', lw=2,
                           label=f'inferred mixture')
        # set upper limit
        if upper_limit != None:
            ax[mapper[g]].set_ylim(0, upper_limit)
        if lower_limit != None:
            ax[mapper[g]].set_ylim(lower_limit, upper_limit)

    # mark characteristic times in the plots
    if plot_characteristictimes != False:
        charl = plot_characteristictimes[0]
        timelc = plot_characteristictimes[1]
        for ill in range(0, len(charl)):
            for a in ax:
                a.text((timelc[ill][0] + timelc[ill][1]) / 2., .99, charl[ill],
                       transform=matplotlib.transforms.blended_transform_factory(a.transData, a.transAxes), color='red')

    plt.tight_layout()
    fig.patch.set_facecolor((0, 0, 0, 0))
    ax[0].legend(fontsize=10)
    ax[num_traces - 1].legend(fontsize=10)

    if plot_comp != False:
        comp_plots(plot_comp,chrom_df, peak_df)

    if plot_output != False:
        fig.savefig(plot_output)

    if plot_show != False:
        plt.show()

    if show_viz == False:
        plt.close()
    return [chrom_df, peak_df, [fig, ax]]

def comp_plots(peak_subset,chrom, peak, ):
    """
    compare multiple chromatograms in one figure
    :param chrom:
    :param peak:
    :return:
    """


def convert_HPLC_Zurich(foldername, raw_files):
    # convert raw files into simple files storing chromatograms only
    raw_files = glob.glob(foldername + '/*.txt')

    os.makedirs(foldername + '/converted', exist_ok=True)
    for i in range(len(raw_files)):
        # Create dataframe from the output file
        raw_data = pd.read_csv(raw_files[i], sep="\t", names=['time_min', 'step', 'intensity_mV'])
        raw_data = raw_data.drop(raw_data.index[range(40)])

        for x in range(len(raw_data)):                                #!!!! original 2400
            raw_data.iloc[x, 2] = raw_data.iloc[x, 2].replace("- ", "-")
        raw_data = raw_data.replace({"'": ""}, regex=True)
        raw_data = raw_data.astype(float)

        # Create a new folder with csv files with only the chromatogram values
        raw_data.to_csv(foldername + '/converted/' + raw_files[i].replace(foldername, '').replace('.txt', '') + '.csv',
                        index=False)


##UPLC Script#######################################################################################
# function to replace apostrophes with empty string
def process_file(file_path):
    # opens file
    with open(file_path, 'r') as file:
        text = file.read()

    # replace apostrophes with empty string
    text = text.replace("'", "")

    # writes object to file
    with open(os.path.splitext(file_path)[0] + '.txt', 'w') as file:
        file.write(text)

    print(f"Processing file {file_path}")


# Path to the folder containing the files
folder_path = "raw"

# Loop through all the files in the folder
for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
    if os.path.isfile(file_path):
        # Call your function on the file
        process_file(file_path)

# Path to the folder containing the calibration files
folder_path = "bin/cal"

# Loop through all the files in the folder
for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
    if os.path.isfile(file_path):
        # Call your function on the file
        process_file(file_path)

########################################
print ('########')

#set variables for analysis of HPLC data
foldername_cal ='cal'       #foldername for calibration
foldername_raw = 'raw'      #foldername for data

peaks = ['glutamate', 'internal standard', 'glutamine', 'citrulline', 'arginine', 'GABA',
         'alanine', 'NH4Cl', 'valine', 'tryptophan', 'isoleucine', 'leucine-phenylalanine', 'ornithine' ]#Set which peaks to quantify

#set retention time for each compound for each round
RT_peaks = [1.433, 1.886, 3.293, 5.940, 6.406, 7.493,
            7.906, 10.020, 12.653, 13.880, 14.186, 14.480, 15.433]

########################################
print ('########')
# convert files for calibration

raw_hplc_files = glob.glob(foldername_cal+'/*.txt')
print(foldername_cal+'/*.txt')
print(raw_hplc_files)
convert_HPLC_Zurich(foldername_cal, raw_hplc_files)

#boundpars=[0.5,0.01,0.0,-0.1,1.6,0.01,2,0.1] #set boundaries of fitting
boundpars=[0,0.03,0,-10,1000,0.03,0.1,10] # [0,0.01,0,-0.1,1000,0.01,0.5,0.1]
#1. min. amplitude of peak height (multiplication of amplitude of detected peak position)
#2. min. position of peak (offset of detected peak position to the left)
#3. min. width of fitted peak
#4. min. value of skew parameter
#5. max. amplitude of peak height (multiplication of amplitude of detected peak position)
#6. max. position of peak (offset of detected peak position to the right)
#7. max. width of fitted peak-> if too high will itefere with peak resolution
#8. max. value of skew parameter


#run and detect peaks

data_files = sorted(glob.glob(foldername_cal+'/converted/*.csv'))
plot_dir = os.getcwd() + '/plots'
os.makedirs(plot_dir, exist_ok=True)

data_chroms, data_peaks, data_plot = batch_process(data_files,show_viz=False, plot_output = plot_dir + '/ plots',  time_window=[0, 23], manual_peak_positions=RT_peaks, backgroundsubstraction=True,backgroundsubstraction_iterations = 50, boundpars=boundpars)

#calculate standard deviation of internal standard
filtered_data = data_peaks[data_peaks['peak_idx'] == 3]   #!!!!!!
area_std = np.std(filtered_data['area'])
area_mean = np.mean(filtered_data['area'])
cv = (area_std / area_mean) * 100

#prints                                             ## display function is specific to IPython -> changed to print()
print("Detected peaks")
print(data_peaks)
print('Internal standard')
print('Standard Deviation:', area_std,'Mean:', area_mean, 'Coefficent of Variation:', cv)
print(data_peaks.info())

#########
print ('########')
for s in data_peaks['sample'].values:
    # Split the file name by underscores
    split = s.split('_')
    sample = split[0].split()[0]
    sample = sample.replace('converted\\','')
    data_peaks.loc[data_peaks['sample'] == s, 'sample'] = sample

#print(data_peaks)

######
print ('########')
data_peaks.to_csv(foldername_cal + '.csv', index=False)
# get list with concentration and peak intensity
concs = []
# Loop through each sample name
for s in data_peaks['sample'].values:
    # Split the file name by underscores
    split = s.split('_')
    # print(split)
    ## Take only the last item that was split by an underscore and remove the mM
    concentration = split[0].split('mM')[0]
    concentration = concentration.replace('converted\\', '')
    ## Add the concentration to the storage list as a float

    concs.append(float(concentration))

# Add the concentration as a new column in the data.
data_peaks.loc[:, 'concentration_mM'] = concs  # warning

####
print ('########')
#adding json directory (not needed in Ipython)
json_dir = os.getcwd() + '/json'
os.makedirs(json_dir, exist_ok=True)


error = 0.05
data_peaks_index = data_peaks.columns.tolist()

#for each expected compound
for compound_i in range(0,len(peaks)):
    data_subset = pd.DataFrame(columns = data_peaks_index)
    idx = 0

    #get subset of peaks which fall in the expected RT interval
    for peak_i in range(0,len(data_peaks)):
        current_peak_RT = data_peaks['retention_time'].iloc[peak_i]
        if (RT_peaks[compound_i] - error) <= current_peak_RT <= (RT_peaks[compound_i] + error):
            data_subset.loc[idx] = list(data_peaks.iloc[peak_i])
            idx = idx + 1

    #check that there are peaks at the expected RT in all calibration samples
    if len(data_subset) == len(data_peaks['sample'].unique()):
        print('Full calibration at retention time ',RT_peaks[compound_i] , ' corresponding to ',peaks[compound_i])
        # perform linear regression for the compound
        output = scipy.stats.linregress(data_subset['area'], data_subset['concentration_mM'])

        # Extract slope, intercept, and r-value from output tuple
        slope = output[0]
        intercept = output[1]
        r_value = output[2]

        # Print results
        print("******Calibration curve for ", peaks[compound_i], "*****")
        print("Slope:", slope)
        print("Intercept:", intercept)
        print("R-squared:", r_value ** 2)

        # Create dictionary to store calibration results
        json_par = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2
        }

        # Serialize dictionary to JSON format
        calibration = json.dumps(json_par, indent=2)

        # Define output file name
        outjson = f"json/calibration_{peaks[compound_i]}.json"

        # Write calibration results to file
        with open(outjson, "w") as jsonFile:
            json.dump(json_par, jsonFile)

        print("Calibration curve saved as:", outjson)

        # Set up a range of areas to plot
        area_range = np.linspace(0, 1.1 * data_subset['area'].max(), 300)

        # Compute the calibration curve for plotting
        fit = intercept + slope * area_range

        # Create a new figure for the plot
        plt.figure()

        # Plot the data points and the calibration curve
        plt.plot(area_range, fit, 'k-', label='fit')
        plt.plot(data_subset['area'], data_subset['concentration_mM'], 'o')

        # Add a legend and axis labels
        plt.legend()
        plt.xlabel('Value [a.u.]')
        plt.ylabel('concentration [mM]')

        # Set the title of the plot to include the peak_idx value
        plt.title('Calibration curve for ' + peaks[compound_i])

        # Save the plot to a file
        plt.savefig('plots/peak_' + peaks[compound_i] + '_calibration.png')

        # Show the plot
        #plt.show()

    elif len(data_subset) > len(data_peaks['sample'].unique()):
        print('error: calibration peak number at retention time ',RT_peaks[compound_i] , ' corresponding to ',peaks[compound_i] ,'higher than expected.')

    elif len(data_subset) < len(data_peaks['sample'].unique()):
        print('error: calibration peak number at retention time ',RT_peaks[compound_i] , ' corresponding to ',peaks[compound_i] ,'lower than expected.')


####
print ('########')

#convert files of measurement

raw_hplc_files = glob.glob(foldername_raw+'/*.txt')
print(raw_hplc_files)
convert_HPLC_Zurich(foldername_raw, raw_hplc_files)

#boundpars defined earlier in script


data_files = sorted(glob.glob(foldername_raw+'/converted/*.csv'))
plot_dir = os.getcwd() + '/plots'

data_chroms, data_peaks, data_plot = batch_process(data_files, show_viz=False, plot_output = plot_dir + '/plots', plot_show = False, time_window=[0, 23],
                                                                  manual_peak_positions=RT_peaks, backgroundsubstraction=True,
                                                                  backgroundsubstraction_iterations = 50, boundpars=boundpars)

print("Detected peaks")
print(data_peaks)
print(data_peaks.info())

####
print ('########')

#calculate standard deviation of internal standard
filtered_data = data_peaks[data_peaks['peak_idx'] == 3] #!!!!!!!
area_std = np.std(filtered_data['area'])
area_mean = np.mean(filtered_data['area'])
cv = (area_std / area_mean) * 100

print('Internal standard')
print('Standard Deviation:', area_std,'Mean:', area_mean, 'Coefficent of Variation:', cv)

####
print ('########')

for s in data_peaks['sample'].values:
    # Split the file name by underscores
    split = s.split('_')
    sample = split[0].split()[0]
    sample = sample.replace('converted\\','')
    data_peaks.loc[data_peaks['sample'] == s, 'sample'] = sample

print(data_peaks)

####
print ('########')
#make result directory
result_dir = os.getcwd() + '/result'
os.makedirs(result_dir, exist_ok=True)


# This code can be used for quantification of each compound in each sample
# go through samples and find specific peaks

samplelist = data_peaks['sample'].unique()

all_unid_peaks = pd.DataFrame(columns = data_peaks.columns.tolist())

# Set which peaks to quantify
#peaks = ['glutamic acid', 'internal standard', 'glutamine', 'alanine', 'citrulline', 'arginine', 'GABA', 'ammonium','valine', 'tryptophan', 'isoleucine', 'leu_phe', 'ornithine']

# for each sample to analyse
for sample in samplelist:
    #initialise empty variables
    compounds = []
    concentration = []
    unidentified_idx = []
    identified_idx = []
    id_output = {}

    #extract information about identified peaks for this sample
    data_selection = data_peaks.loc[data_peaks['sample'] == sample]

    # initialise empty dataframes with same indices as data_selection
    data_sel_index = data_selection.columns.tolist()
    identified_peak_data = pd.DataFrame(columns = data_sel_index)
    unidentified_peak_data = pd.DataFrame(columns = data_sel_index)

    # error margin in RT allowed for the identification of each peak
    error = 0.05

    #list of compounds for which a calibration curve has been determined
    cal_curves = glob.glob('json/'+'/*.json')
    cal_comp = []
    for cal in range(0,len(cal_curves)):
        cal_comp.append(cal_curves[cal].replace('json/calibration_', '').replace('.json', ''))


    # for each peak in this sample:
    for i in range(0, len(data_selection)):
        current_RT = data_selection.iloc[i]['retention_time']

        #check if peak falls in range of RT of expected compounds +- error margin defined above
        for j in range(0, len(RT_peaks)):
            if peaks[j] in cal_comp:
                if (RT_peaks[j] - error) <= current_RT <= (RT_peaks[j] + error):
                    compounds.append(peaks[j])
                    identified_idx.append(i)
                    identified_peak_data.loc[len(compounds)-1] = list(data_selection.iloc[i]) #not the most elegant solution

                    #get calibration curve information
                    with open("json/calibration_" + peaks[j] + ".json", "r") as jsonFile:
                        cal_data = json.load(jsonFile)
                        slope = cal_data["slope"]
                        intercept = cal_data["intercept"]

                    #using area of current peak and information from calibration curve, calculate concentration of compound
                    concentration.append((data_selection.iloc[i]['area'] - intercept)/np.abs(slope))
                    break

        # if peak does not fall in range -> mark as unidentified
        if i not in identified_idx:
            unidentified_idx.append(i)

    #extract peak information about unidentified peaks
    for k in range(0, len(unidentified_idx)):
        unidentified_peak_data.loc[k] = list(data_selection.iloc[unidentified_idx[k]])

    id_output = {'compound': compounds, 'concentration (mM)': concentration}
    id_output = pd.DataFrame(data=id_output)  #####this is far from optimal
    id_output = pd.concat([id_output,identified_peak_data], axis = 1)


    # Save the data_peaks dataframe for this sample to a csv file
    id_output.to_csv('result/' + sample + '_output.csv', index=False)
    unidentified_peak_data.to_csv('result/' + sample + '_unidentified_peaks.csv', index=False)