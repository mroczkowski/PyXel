import numpy as np

def rotate_point(x0, y0, x, y, angle):
    """Rotate point (x,y) counter-clockwise around (x0,y0)."""
    x_rot = x0 + x * np.cos(angle) - y * np.sin(angle)
    y_rot = y0 + x * np.sin(angle) + y * np.cos(angle)
    return (x_rot, y_rot)

def bin_pix2arcmin(bin_r, bin_width, raw_cts, src, err_src, bkg, err_bkg,
    net, err_net, pix2arcmin):
    '''blah blah'''
    bin_r = bin_r * pix2arcmin * 60.
    bin_width = bin_width * pix2arcmin * 60.
    src = src / pix2arcmin**2 / 3600.
    err_src = err_src / pix2arcmin**2 / 3600.
    bkg = bkg / pix2arcmin**2 / 3600.
    err_bkg = err_bkg / pix2arcmin**2 / 3600.
    net = net / pix2arcmin**2 / 3600.
    err_net = err_net / pix2arcmin**2 / 3600.
    return (bin_r, bin_width, raw_cts, src, err_src, bkg, err_bkg, net, err_net)

def get_bkg_exp(bkg_img, exp_img):
    if bkg_img is None:
        bkg_img_data = np.zeros(np.shape(counts_img.data))
        bkg_norm_factor = 1
    else:
        bkg_img_data = bkg_img.data
        if 'BKG_NORM' in bkg_img.hdr:
            bkg_norm_factor = bkg_img.hdr['BKG_NORM']
        else:
            print(InfoMessages('003'))
            bkg_norm_factor = 1
    if exp_img is None:
        exp_img_data = np.ones(np.shape(counts_img.data))
    else:
        exp_img_data = exp_img.data
    return bkg_img_data, bkg_norm_factor, exp_img_data

def merge_subpixel_bins(edges):
    new_edges = [edges[0]]
    start_edge = edges[0]
    for edge in edges[1:]:
        if edge - edge_start < 1:
            continue
        else:
            new_edges.append(edge)
            start_edge = edge
    return new_edges

def get_edges(max_r, islog):
    if not islog:
        nbins = max_r
        # Below, nbins+1 is used because the code gets edges, not
        # bin centers. For nbins there will be nbins+1 edges
        return np.linspace(0., max_r, nbins + 1)
    else:
        # If the bins are simply distributed logarithmically such that the
        # smallest bin has a width of at least 1 pixel, one ends up with an
        # unreasonably small number of bins. So, instead, the radial range
        # is divided into many logarithmically-scaled bins, some with
        # sub-pixel width, and then the merging will take care of them
        # later. The end result is a compromise between log-scaling and
        # getting one's time worth of bins.
        nbins = 500.
        min_r = 1.   # to avoid log(0)
        # Below, nbins+1 is used because the code gets edges, not
        # bin centers. For nbins there will be nbins+1 edges.
        edges = np.logspace(min_r, max_r, nbins + 1)
    # Inserts the 0 edge back into the array of edges.
    edges = list(np.insert(edges, 0, 0.))
    return merge_subpixel_bins(edges)

def get_data_for_chi(profile, minrange, maxrange):
    nbins = len(profile)
    r = np.array([profile[i][0] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    w = np.array([profile[i][1] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    net = np.array([profile[i][7] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    net_err = np.array([profile[i][8] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    return nbins, r, w, net, net_err

def get_data_for_cash(profile, minrange, maxrange):
    nbins = len(profile)
    r = np.array([profile[i][0] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    w = np.array([profile[i][1] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    raw_cts = np.array([profile[i][2] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    sb_to_counts_factor = np.array([profile[i][9] for i in range(nbins) if minrange <= profile[i][0] <= maxrange])
    return nbins, r, w, raw_cts, sb_to_counts_factor

def call_model(func_name):
    model = getattr(model_defs, func_name)
    if not model:
        raise Exception("Model %s is not implemented." % func_name)
    else:
        return model
