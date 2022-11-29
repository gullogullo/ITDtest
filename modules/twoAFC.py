class TwoAFC:
    """ Base class for 2I-2AFC psychometric method
    """
    def __init__(self, start_itd=40, initial_step=2, first_downup_step= 1.414,
        second_downup_step = 1.189, total_reversals = 6):
        """
        Method arguments:
            Three down - one up:
                starting ITD -- 40*10^-6s 
                initial step -- factor 2
                first reversal down-up -- 1.414
                second reversal down-up -- 1.189
                total reversals to stop -- 6 TODO check "at minimum step size"????
        """

        # METHOD SETTINGS
        self.start_itd = start_itd 
        self.initial_step = initial_step 
        self.first_downup_step = first_downup_step
        self.second_downup_step = second_downup_step 
        self.total_reversals = total_reversals
        self.reversals = 0
        self.downup_reversals = 0
        self.answers = {}
        self.answers['ITD'] = []  
        self.answers['Label'] = []