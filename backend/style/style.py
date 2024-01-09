from backend.style.color import Color


class Style:
    """Class containing various OpenCV styles.

    This class contains various OpenCV styles used in the program.
    """
    FACE_BOX_STYLE = {'color': Color.GREEN, 'thickness': 2}
    EYE_BOX_STYLE = {'color': Color.GREEN, 'thickness': 1}
    ROI_BOX_STYLE = {'color': Color.YELLOW, 'thickness': 1}
