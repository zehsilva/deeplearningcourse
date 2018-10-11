from torch.utils.data import Dataset
import datapreparation as datp


class pianoroll_dataset_batch(Dataset):
    """Face Landmarks dataset."""
    """
    TODO: load from dir without loading the whole dataset (for larger dataset)
    """
    def __init__(self, root_dir, transform=None, name_as_tag=True):
        """
        Args:
            root_dir (string): Directory with all the csv
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        if(name_as_tag):
            self.tags =  datp.load_all_dataset_names(self.root_dir)
        self.data = datp.load_all_dataset(self.root_dir)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        return  self.data[idx]
    
    def set_tags(self,lst_tags):
        self.tags = lst_tags
        
    def view_pianoroll(self,idx):
        datp.visualize_piano_roll(self[idx])