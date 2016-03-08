import numpy as np
import matplotlib.pyplot as plt
from aux import rotate_point, bin_pix2arcmin, call_model, get_bkg_exp
from SurfMessages import InfoMessages

class Region(object):

    def get_bin_vals(self, counts_img_data, bkg_img_data, bkg_norm_factor,
        exp_img_data, pixels_in_bin, only_counts=False, only_net_cts=False):
        """Calculate the number of counts in a bin."""
        src = 0
        bkg = 0
        err_src = 0
        err_bkg = 0
        exp = 0
        npix = len(pixels_in_bin)
        raw_cts = 0
        
        for pixel in pixels_in_bin:
            if exp_img_data[pixel[0], pixel[1]] != 0:
                exp_val = exp_img_data[pixel[0], pixel[1]]
                # In case the profile needed is a counts profile...
                if only_counts:
                    exp_val = 1.
                raw_cts += counts_img_data[pixel[0], pixel[1]]
                src += counts_img_data[pixel[0], pixel[1]] / exp_val
                bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val \
                    / bkg_norm_factor
                err_src += counts_img_data[pixel[0], pixel[1]] / exp_val**2
                err_bkg += bkg_img_data[pixel[0], pixel[1]] / exp_val**2 \
                    / bkg_norm_factor**2
                exp += exp_val

        net = src - bkg

        if only_net_cts:
            return net
        else:
            net = net / npix
            src = src / npix
            bkg = bkg / npix
            err_net = np.sqrt(err_src + err_bkg) / npix
            err_src = np.sqrt(err_src) / npix
            err_bkg = np.sqrt(err_bkg) / npix
            exp /= npix
            return raw_cts, src, err_src, bkg, err_bkg, net, err_net

    def profile(self, counts_img, bkg_img, exp_img,
        min_counts=100, only_counts=False, islog=True):
        """Generate count profiles.

        The box is divided into bins based on a minimum number of counts or a
        minimum S/N. The box is divided into bins starting from the bottom up,
        where the bottom is defined as the bin starting at the lowest row in
        the nonrotated box. E.g., if the box is rotated by 135 deg and it
        can be divided into three bins, then the bins will be distributed as:

                                 x
                               x   x
                             x       x
                           x    1st    x
                         x   x           x
                       x       x    bin    x
                     x    2nd    x       x
                   x   x           x   x
                 x       x    bin    x
               x    3rd    x       x
                 x           x   x
                   x    bin    x
                     x       x
                       x   x
                         x

        If a background map is not read in, then the background is set to zero.
        If an exposure map is not read in, then the exposure is set to one
        everywhere. However, if point source exclusion is important, then an
        exposure map is necessary; regions with zero exposure will be ignored,
        so they need to be masked in the exposure map. Removing point sources
        from the exposure map with dmcopy can be very slow if the full field of
        view is not included in the region file. So to quicken things up, the
        region file should look something like:

                field()
                -ellipse(......)
                -circle(......)

        The function returns a list of tuples of the form:
        (bin radius, bin width, source counts, source counts uncertainty,
        background counts, background counts uncertainty, net counts,
        net counts uncertainty)
        """
        bkg_img_data, bkg_norm_factor, exp_img_data = \
            get_bkg_exp(bkg_img, exp_img)
        pix2arcmin = counts_img.hdr['CDELT2']

        bins = self.merge_bins(counts_img, bkg_img, exp_img, min_counts, islog)

        profile = []
        print('PROFILE:')
        for current_bin in bins:
            raw_cts, src, err_src, bkg, err_bkg, net, err_net = \
                self.get_bin_vals(
                    counts_img.data, bkg_img_data, bkg_norm_factor,
                    exp_img_data, current_bin[2], only_counts=only_counts)
            bin_radius = (current_bin[0] + current_bin[1]) / 2.
            bin_width = current_bin[1] - bin_radius
            bin_values = (bin_radius, bin_width, raw_cts, src, err_src,
                          bkg, err_bkg, net, err_net)
            bin_values = bin_pix2arcmin(bin_values, pix2arcmin)
            profile.append(bin_values)
            print(bin_values)
        return profile

    def counts_profile(self, counts_img, bkg_img, exp_img,
        min_counts=100, only_counts=True, islog=True):
        """some docstring"""
        return self.profile(counts_img, bkg_img, exp_img,
            min_counts, only_counts=True, islog=islog)

    def sb_profile(self, counts_img, bkg_img, exp_img,
        min_counts=100, only_counts=False, islog=True):
        """some docstring"""
        return self.profile(counts_img, bkg_img, exp_img,
            min_counts, only_counts=False, islog=islog)

    def plot_profile(self, profile, xlog=True, ylog=True,
        xlims=None, ylims=None, xlabel=None, ylabel=None,
        model_name=None, model=None):
        """Plot profile and (optional) fitted model.

        Plots the net count profile (with error bars) and the background
        profile (step function without error bars - the uncertainties are
        difficult to get from a renormalized background image without knowing
        the normalization). The plotting can easily be done without the use of
        this routine, by just calling count_profile to get the data. This would
        allow for more customization than this routine provides.
        """
        nbins = len(profile)

        r = np.array([profile[i][0] for i in range(nbins)])
        r_err = np.array([profile[i][1] for i in range(nbins)])

        bkg = np.array([profile[i][5] for i in range(nbins)])
        net_cts = np.array([profile[i][7] for i in range(nbins)])
        err_net_cts = np.array([profile[i][8] for i in range(nbins)])

        plt.scatter(r, net_cts, c="black", alpha=0.85, s=35, marker="s")
        plt.errorbar(r, net_cts, xerr=r_err, yerr=err_net_cts,
                     linestyle="None", color="black")
        plt.step(r, bkg, where="mid", linewidth=3)

        plt.rc('text', usetex=False)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        plt.grid(True)

        if xlims is not None:
            plt.xlim([xlims[0], xlims[1]])
        else:
            plt.xlim([0.01, np.max(r + r_err)])

        if ylims is not None:
            plt.ylim([ylims[0], ylims[1]])

        if xlog:
            plt.semilogx()

        if ylog:
            plt.semilogy()

        if model is not None and model_name is not None:
            if not model_name:
                raise Exception("No model is defined.")
            else:
                evaluated_model = model.evaluate(r)
                plt.plot(r, evaluated_model, color="r", linewidth=3, alpha=0.75)

        plt.show()
