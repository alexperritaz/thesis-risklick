# -*- coding: utf-8 -*-
import re
import lxml.etree as ET

from utils import walk_files, progress_bar, read_csv, write_csv

DELIMITER = '|||'

class Convert:
    
    def __init__(self, source_directory, source_features, whole_document):        
        self.source_directory = source_directory
        self.features = read_csv(source_features)
        self.data = []
        self.process = self.as_whole_document if whole_document else self.as_features
        pass
    
    def process(self):
        self.process()
        pass
    
    # Proceed to data conversion from XML to CSV
    def as_features(self):
        filepaths = walk_files(self.source_directory)
        for filepath in progress_bar(filepaths, "Processing : "):
            self.new_row(filepath)
        self.data.insert(0, self.features)
        pass
    
    # Process whole study as text
    def as_whole_document(self):
        filepaths = walk_files(self.source_directory)
        for filepath in progress_bar(filepaths, "Processing : "):
            xml_doc_str = ET.tostring(ET.parse(filepath)).decode()
            self.data.append([xml_doc_str])
        self.data.insert(0, 'Clinical_Trials')        

    # Create a new data row
    def new_row(self, filepath):
        row = self.create_row(filepath)
        row = self.update_row(filepath, row)
        self.data.append(row)
        pass
    
    # Create empty row with matching identifier
    def create_row(self, filepath):        
        row = [-1] * len(self.features)
        row[0] = filepath.split('/')[-1].split('.')[0]        
        return row
    
    # Update current row with new data entry
    def update_row(self, filepath, row):        
        tree = ET.parse(filepath)
        root = tree.getroot()                    
        for node in root.iter():
            text = re.sub('\s+', ' ', node.text.strip())
            if text:
                featurename = self.format_path(tree, node)
                pos = self.features.index(featurename)
                row[pos] = self.add_content(row[pos], text)
        return row
    
    # Remove '[#]' to keep general features
    def format_path(self, tree, node):
        path = tree.getpath(node)
        path = re.sub("[\(\[].*?[\)\]]", "", path)
        path = path.lower()
        return path
    
    # Add data to the current default cell
    def add_content(self, cell, text):        
        if cell == -1: 
            return text
        else: 
            return self.update_content(cell, text)
    
    # Add data to the current cell with already existing data
    def update_content(self, cell, text):
        if text not in cell.split(DELIMITER):
            cell += DELIMITER + text        
        return cell
    
    # Save file
    def save(self, dest_file):
        write_csv(dest_file, self.data, True)
        pass