from functools import wraps

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from cycler import cycler
import h5py

########################
# Helper functions
########################

def optional_figax(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, fig = None, ax = None, 
                w: float = 8.0, h: float | None = None, **kwargs):

        h = 0.4*w if h is None else h
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(w, h))
        return func(*args, fig=fig, ax=ax, **kwargs)
    return wrapper

def grid(n: int, nrows: int = 1, *, w: float = 3.0, h: float | None = None,
         sharexy: bool = False,**kwargs):

    h = w if h is None else h
    ncols = np.ceil(n / nrows)
    figsize = (w * ncols, h * nrows)
    if sharexy:
        kwargs['sharex'] = True
        kwargs['sharey'] = True

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axs.flatten()

def read_from_file(filepath: str):
    """Read a class and associated data from file, without loading it into RAM.

    Contiguous numeric datasets are returned as read-only np.memmap arrays. Chunked
    datasets cannot be memory mapped (their chunks are scattered through the file), so
    they are returned as lazy h5py.Dataset handles that only read the slices you index.
    Small/scalar/non-numeric datasets are read eagerly.

    Parameters:
        filepath: Path to the file containing both raw data and the information needed
            to reinitialize our class

    Returns:
        data_dict: Dictionary of data that was passed to *merged* write_to_file at the time
    """

    def load_dataset(item):
        offset = item.id.get_offset()  # None unless contiguous and allocated
        if item.size == 0 or item.ndim == 0 or item.dtype.kind not in "biufc":
            return item[()]
        if offset is not None:
            return np.memmap(filepath, mode="r", dtype=item.dtype, shape=item.shape,
                             offset=offset, order="C")
        return item

    def load_hdf5_dict(group):
        """Recursively load HDF5 group/dataset structure back into a dict"""
        data_dict = {}

        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # It's a nested dict, recurse into it
                data_dict[key] = load_hdf5_dict(item)
            elif isinstance(item, h5py.Dataset):
                data_dict[key] = load_dataset(item)

        return data_dict

    # Not using a context manager: the file must stay open for the lazy h5py.Dataset
    # handles. It is closed once they are garbage collected.
    f = h5py.File(filepath, "r")
    return load_hdf5_dict(f)

########################
# FloquetPlot class
########################

class FloquetPlot:
    def __init__(self, directory, zeta_value, ng_values, use_tex=False, fontsize=12):
        self.directory = directory
        self.zeta_value = zeta_value
        self.data = {} # Key: idx, Value: data

        self.sweep_vals = ng_values # in this case: ng values
        self.sweep_idxs = range(len(ng_values))
        self.num_sweep_idxs = len(self.sweep_idxs)

        # Load data from idx = 0 file 
        # TODO: Doing this manually for now
        filepath = f"{self.directory}/zeta_{self.zeta_value:.4f}_ng_{self.sweep_vals[0]:.4f}.h5"
        data_dict = read_from_file(filepath)

        self.xi_sq  = data_dict['xi_sq']    # Note: is an array
        self.omega_d = data_dict['omega_d'] # Note: is an array

        # Params and state labels
        self.hilbert_dim = data_dict['avg_excitation'].shape[-1]
        self.lookup = np.arange(self.hilbert_dim)
        
        self._plot_style(use_tex=use_tex)
        self.fontsize = fontsize

    def _plot_style(self, use_tex=False):

        # Style+Color cycler
        color_cycler = cycler(plt.rcParams["axes.prop_cycle"])
        ls_cycler = cycler(ls=["-", "--", "-.", ":"])
        alpha_cycler = cycler(alpha=[1.0, 0.6, 0.2])
        lw_cycler = cycler(lw=[1.5,2.5,3.5,4.5])
        self.color_ls_alpha_cycler = lw_cycler * alpha_cycler * ls_cycler * color_cycler 

        plt.rc('xtick', direction='in') 
        plt.rc('xtick.major', width=0.75)
        plt.rc('xtick.minor', width=0.75)
        plt.rc('ytick', direction='in')
        plt.rc('ytick.major', width=0.75)
        plt.rc('ytick.minor', width=0.75)
        
        plt.rc('axes', unicode_minus = False, linewidth=0.5)
        plt.rc('figure.constrained_layout', use=True)

        if use_tex:
            plt.rc('font', family = 'serif')
            plt.rc('figure', dpi = 80)
            plt.rc('savefig', dpi = 100)
            plt.rc('text', usetex=True)
            plt.rc('mathtext', fontset = 'custom')

    def get_data(self, idx):
        if not isinstance(idx, (int, np.integer)) or idx >= self.num_sweep_idxs:
            raise ValueError(f"Invalid idx: {idx}")

        if idx in self.data.keys():
            return self.data[idx]
        
        filepath = f"{self.directory}/zeta_{self.zeta_value:.4f}_ng_{self.sweep_vals[idx]:.4f}.h5"
        data_dict = read_from_file(filepath)

        xi_sq  = data_dict['xi_sq']    # Note: is an array
        omega_d = data_dict['omega_d'] # Note: is an array
        data = {
            'displaced_state_overlaps' : data_dict["displaced_state_overlaps"],
            'quasienergies'  : data_dict["quasienergies"],
            'avg_excitation' : data_dict["avg_excitation"],
        }
        
        # Validate data
        assert np.allclose(xi_sq,self.xi_sq), \
                f"xi_sq does not match at idx={idx}"
        assert np.allclose(omega_d,self.omega_d), \
                f"omega_d does not match at idx={idx}"
        
        self.data[idx] = data
        return data

    @optional_figax
    def landscape(self, *, state_idx, idx=None,
                style='jaya_xu',
                omega_d_idx=-1, omega_d_lims = None,
                fig=None, ax=None, num_x_ticks=None, xticklabels=None,
                show_colorbar=True, colorbar_pad=0.0,
                interpolation="none"):
        """
        Plot landscape plot. 
        Inputs:
            state_idx    : 0 = |g>, 1 = |e>
            idx      : Index (of the ng or flux value) to plot
            style        : jaya_xu or blais
            omega_d_idx  : Drive frequency where we take a linecut
            omega_d_lims : Drive frequency limits
            fig, ax      : Figure and axis objects to plot on
        """
        if state_idx == 0:
            cmap = "Blues"
        elif state_idx == 1:
            cmap = "Reds"
        else:
            raise ValueError

        # Either take only data from one idx or average over data from multiple idxs
        if idx is None:
            idx = list(range(self.num_sweep_idxs))

        if style == 'jaya_xu':
            if isinstance(idx, (float, np.integer)):
                idx = int(idx)

            if isinstance(idx, int):
                data = self.get_data(idx)['displaced_state_overlaps'][:,:,state_idx].T ** 2
            elif isinstance(idx, list):
                data = sum([self.get_data(_idx)['displaced_state_overlaps'][:,:,state_idx] ** 2 
                                for _idx in idx]).T / len(idx)
            else:
                raise ValueError(f"Invalid idx: {idx}")

            data = np.clip(1 - data, 0.0, 0.2)

        elif style == 'blais':
            if isinstance(idx, (float, np.integer)):
                idx = int(idx)

            if isinstance(idx, int):
                data = self.get_data(idx)['avg_excitation'][...,state_idx].T
            elif isinstance(idx, list):
                data = sum([self.get_data(_idx)['avg_excitation'][...,state_idx].T for _idx in idx]) / len(idx)
            else:
                raise ValueError(f"Invalid idx: {idx}")
            data = np.clip(data, 0, 3)

        else:
            raise ValueError('Unrecognized style. Valid ones are jaya_xu and blais')

        # X and Y axis
        xticks = self.omega_d
        yticks = self.xi_sq / 2
        num_x_pts = len(xticks)
        num_y_pts = len(yticks)

        # Plot data
        im = ax.imshow(data, origin="lower", cmap=cmap, interpolation=interpolation, aspect='auto')

        # Linecut
        if omega_d_idx >= 0:
            ax.axvline(omega_d_idx, color="grey", ls="--")

        # Title and axes
        if num_x_ticks is not None:
            xticklabel_locations = np.linspace(0, num_x_pts - 1, num_x_ticks, dtype=int, endpoint=True)
            xticklabels = np.around(xticks[xticklabel_locations], decimals=2)
        elif xticklabels is not None:
            xticklabels = np.asarray(xticklabels)
            # Nearest grid point; exact float equality misses e.g. 7.000000000000001 from linspace
            xticklabel_locations = np.array([np.argmin(np.abs(xticks - p)) for p in xticklabels])
        else:
            xticklabels = np.array([xticks[0]])
            xticklabels = np.concatenate([
                xticklabels, 
                np.arange(np.ceil(xticks[0]).astype(int), np.floor(xticks[-1]).astype(int))
            ])
            xticklabels = np.concatenate([
                xticklabels,
                np.array([xticks[-1]])
            ])
            xticklabel_locations = np.array([np.argmin(np.abs(xticks - p)) for p in xticklabels])

        ax.set_xticks(xticklabel_locations)
        ax.set_xticklabels(xticklabels.astype(str), fontsize=self.fontsize)
        yticklabel_locations = np.linspace(0, num_y_pts - 1, 3, dtype=int, endpoint=True)
        ax.set_yticks(yticklabel_locations)
        ax.set_yticklabels(np.array(np.around(yticks[yticklabel_locations], decimals=2), dtype=str),
                           fontsize=self.fontsize)
        ax.set_ylabel(r"$\Delta_{AC}/\alpha_q$", fontsize=self.fontsize)
        ax.set_xlabel(r"$\omega_d/\omega_p$", fontsize=self.fontsize)

        if omega_d_lims is not None:
            ax.set_xlim(*omega_d_lims)

        if show_colorbar:
            colorbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='max', 
                                    pad=colorbar_pad, shrink=0.9)
                                    
            if style == 'jaya_xu':
                colorbar.set_label(fr"$\Theta({state_idx}_t)$", fontsize=self.fontsize)
            elif style == 'blais':
                colorbar.set_label(fr"$\langle n_t \rangle$", fontsize=self.fontsize)
            colorbar.ax.tick_params(direction='out', labelsize=self.fontsize)

            ticks = np.round(np.linspace(np.min(data), np.max(data), 3), decimals=3)
            colorbar.set_ticks(ticks)

        return data,im

    @optional_figax
    def blais_branch(self, *,idx, omega_d_idx, lookup_idxs=None, fig=None, ax=None):
        """
        Plot Blais-style branch analysis plot. 
        Inputs:
            idx      : Index of the ng value to plot
            omega_d_idx : Drive frequency where we take a linecut
            lookup_idxs : List of indices to highlight
            fig, ax     : Figure and axis objects to plot on
        """
        if omega_d_idx < 0:
            return

        if lookup_idxs is None:
            lookup_idxs = range(len(self.lookup))

        omega_d = self.omega_d[omega_d_idx]
        if isinstance(idx, (float, np.integer)):
            idx = int(idx)

        if isinstance(idx, int):
            avg_excitation = self.get_data(idx)['avg_excitation'][omega_d_idx, :, :]
        elif isinstance(idx, list):
            avg_excitation = sum([self.get_data(_idx)['avg_excitation'][omega_d_idx, :, :] for _idx in idx]) / len(idx)
        else:
            raise ValueError(f"Invalid idx: {idx}")
        
        zorder = 999
        sty_cycler = iter(cycler(self.color_ls_alpha_cycler))

        for idx, label in enumerate(self.lookup):
            if idx in lookup_idxs:
                sty = next(sty_cycler)
                if isinstance(label, np.int64) or isinstance(label, int):
                    label = fr"${label}$"
                else:
                    label = fr"${idx} = ({label[0]}_t, {label[1]}_r)$"
                sty['zorder'] = zorder
                sty['lw'] = 2.5
                zorder -= 1
            else:
                sty = {'color' : "#D9D9D9", 'ls':"-", 'alpha':0.6}
                label = None
                sty['zorder'] = idx

            ax.plot(self.xi_sq/2, avg_excitation[:, idx], label=label, **sty)

        ax.set_xlabel(r"$\Delta_{AC}/\alpha_q$", fontsize = self.fontsize)
        ax.set_ylabel("Average excitation", fontsize = self.fontsize)
        ax.set_title("Branch analysis\n" + fr"($\omega_d/\omega_pi$ = {omega_d/(2*np.pi):.3f} GHz)",
                     fontsize = self.fontsize)

    @optional_figax
    def qenergy(self, *, idx, omega_d_idx, lookup_idxs=None, qenergy_lims=None, 
                fig=None, ax=None):
        """
        Plot quasienergies plot. 
        Inputs:
            idx          : Index of the ng value to plot
            omega_d_idx  : Drive frequency where we take a linecut
            lookup_idxs  : List of indices to highlight
            qenergy_lims : Y-limits for the quasienergies plot
            fig, ax      : Figure and axis objects to plot on
        """
        if omega_d_idx < 0:
            return

        if lookup_idxs is None:
            lookup_idxs = range(len(self.lookup))

        omega_d = self.omega_d[omega_d_idx]
        if isinstance(idx, (float, np.integer)):
            idx = int(idx)

        if isinstance(idx, int):
            qenergies = self.get_data(idx)['quasienergies'][omega_d_idx, :, :]
        elif isinstance(idx, list):
            qenergies = sum([self.get_data(_idx)['quasienergies'][omega_d_idx, :, :] for _idx in idx]) / len(idx)
        else:
            raise ValueError(f"Invalid idx: {idx}")
    
        quasi_energy_max_min = omega_d/(4*np.pi)
        if qenergy_lims is None:
            qenergy_lims = (-1.1*quasi_energy_max_min, 1.1*quasi_energy_max_min)
        zorder = 999
        sty_cycler = iter(cycler(self.color_ls_alpha_cycler)) 

        # Helper to unwrap the quasienergy modulo L
        def unwrap_modulo(y, L = quasi_energy_max_min):
            y_unwrapped = np.copy(y)
            for i in range(1, len(y)):
                delta = y[i] - y[i - 1]
                if delta > 0.9*L:
                    y_unwrapped[i:] -= 2 * L
                elif delta < -0.9*L:
                    y_unwrapped[i:] += 2 * L
            return y_unwrapped

        for idx, label in enumerate(self.lookup):
            if idx in lookup_idxs:
                sty = next(sty_cycler)
                if isinstance(label, np.int64) or isinstance(label, int):
                    label = fr"${label}$"
                else:
                    label = fr"${idx} = ({label[0]}_t, {label[1]}_r)$"
                sty['zorder'] = zorder
                sty['lw'] = 2.5
                zorder -= 1
            else:
                sty = {'color' : "#D9D9D9", 'ls':"-", 'alpha':0.6}
                label = None
                sty['zorder'] = idx

            # Copy the quasienergy trace to the top and bottom of the plot
            for trace_copy in [-1,0,1]:
                ax.plot(
                    self.xi_sq/2,
                    (2*trace_copy*quasi_energy_max_min) + unwrap_modulo(qenergies[:,idx]/(2*np.pi)),
                    label= label if trace_copy==0 else None,
                    **sty,
                )

        ax.set_xlabel(r"$\Delta_{AC}/\alpha_q$", fontsize = self.fontsize)
        ax.set_ylabel("Quasienergy [GHz]", fontsize = self.fontsize)
        ax.set_ylim(*qenergy_lims)
        ax.axhline(y=quasi_energy_max_min, color='black', linestyle='--', linewidth=2, zorder=1000)
        ax.axhline(y=-quasi_energy_max_min, color='black', linestyle='--', linewidth=2, zorder=1000)
        ax.set_title("Quasienergies\n" + fr"($\omega_d/\omega_p$ = {omega_d/(2*np.pi):.3f} GHz)",
                     fontsize = self.fontsize)

    def landscape_and_cross_section(self, *, omega_d_idx,
                                    style = 'jaya_xu', idx=0,
                                    omega_d_lims = None, num_x_ticks=7,
                                    qenergy_lims=None,        
                                    lookup_idxs=None, save=False, figsize=(8.0,9.0),
                                    fig=None, axs=None, title=""):
                
        if (fig is None) and (axs is None):
            if omega_d_idx >= 0:
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.05, figure=fig)
                axs = [fig.add_subplot(gs[0, :]),  # First row spans both columns
                       fig.add_subplot(gs[1, :]),  # Second row spans both columns
                       fig.add_subplot(gs[2, 0]),  # Third row, first column
                       fig.add_subplot(gs[2, 1])]  # Third row, second column
            else:
                fig, axs = plt.subplots(2,1,figsize=(figsize[0], 0.7*figsize[1]))
                axs = axs.flatten()

        elif (fig is not None) and (axs is not None):
            if omega_d_idx >= 0:
                assert len(axs)==4, f"Provided axs of length {len(axs)}. Must be of length 4"
            else:
                assert len(axs)==2, f"Provided axs of length {len(axs)}. Must be of length 2"
        
        else:
            raise ValueError("Provide both fig and axs, or neither")

        self.landscape(state_idx=0, idx=idx, style=style, omega_d_idx=omega_d_idx, 
                       fig=fig, ax=axs[0],
                       omega_d_lims=omega_d_lims, num_x_ticks=num_x_ticks, colorbar_pad=-0.05)
        self.landscape(state_idx=1, idx=idx, style=style, omega_d_idx=omega_d_idx, 
                       fig=fig, ax=axs[1],
                       omega_d_lims=omega_d_lims, num_x_ticks=num_x_ticks, colorbar_pad=-0.05)

        axs[0].set_xticklabels([])
        axs[0].set_xlabel(None)

        if omega_d_idx >= 0:
            self.blais_branch(idx=idx, omega_d_idx=omega_d_idx, lookup_idxs=lookup_idxs, 
                            fig=fig, ax=axs[2])
            self.qenergy(idx=idx, omega_d_idx=omega_d_idx, lookup_idxs=lookup_idxs, 
                        qenergy_lims=qenergy_lims, fig=fig, ax=axs[3])

            # Share a single legend for the last two plots
            handles, labels = axs[2].get_legend_handles_labels()
            axs[3].legend(handles, labels, bbox_to_anchor=(1, 1))

        if title != "":
            fig.suptitle(title, fontsize = 1.5 * self.fontsize)

        if save:
            plt.savefig(f"{self.directory}/omega_d_idx={omega_d_idx}.pdf", bbox_inches="tight")
