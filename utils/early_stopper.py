class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_monitored_metric = float('-inf')

    def early_stop(self, monitored_metric):
        if monitored_metric > (self.max_monitored_metric + self.min_delta):
            self.max_monitored_metric = monitored_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarlyStopper: The monitored metric did not improve for {self.patience} epochs with delta={self.min_delta}. Early Stopping criterion reached.")
                return True
        return False
