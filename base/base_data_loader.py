class BaseDataLoader():
    def __init__(self, train_loader, val_loader):
        self.train_loader(train_loader)
        self.val_loader(val_loader)
