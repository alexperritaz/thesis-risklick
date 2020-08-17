# -*- coding: utf-8 -*-
import shutil
from utils import walk_files, progress_bar

class Structure():

    # Goes through each directory and subdirectories
    def from_subdirectories(self, src_directory, dst_directory):
        filepaths = walk_files(src_directory)
        for path in progress_bar(filepaths, "Processing files : "):
            shutil.copy2(path, dst_directory)
            
    # Unused - To be implemented if needed
    def from_markup(self, src_directory, dst_directory):
        pass