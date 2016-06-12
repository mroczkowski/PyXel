import textwrap as tw

def remove_whitespace(text):
    return " ".join(text.split())

def ErrorMessages(error_number):
    errors = {
        '001': '''Too few net counts in the region.
            Enlarge the region or lower the minimum
            count threshold.''',
        '002': '''Currently only region files with one
            region defined in image coordinates are supported.'''
    }
    return tw.fill(remove_whitespace(errors[error_number]), 80)

def InfoMessages(info_number):
    info = {
        '003': '''WARNING: Assuming a background normalization factor of 1.
            The background normalization factor can be changed by defining
            the keyword BKG_NORM in the header of the background image.
            To correctly calculate background/net uncertainties, the background
            image provided should have integer values (unnormalized counts).'''
    }
    return tw.fill(remove_whitespace(info[info_number]), 80)
