# -*- coding: utf-8 -*-
import pandas as pd
import lxml.etree as ET

DELIMITER = '|'

class MeSH:
    
    def __init__(self):
        pass
    
    # Extract all mesh terms
    def extract_all(self, src_file):
        # Variables
        tree = ET.parse(src_file)
        root = tree.getroot()
        # Data to save
        self.identifiers = []
        self.data = []
        # Go through all description records
        for record in root:
            # Retrieve reference number of mesh term 
            ref_numbers = record.xpath('./TreeNumberList//TreeNumber/text()')
            # Retrieve mesh name 
            name = record.xpath('./DescriptorName/String/text()')[0]
            self.identifiers.append(name)
            # For each term - multiple reference numbers
            self.data.append(DELIMITER.join(ref_numbers))
        pass    
    
    # Extract mesh terms according to depth specified
    def extract_level(self, src_file, tree_depth, segment_len):
        # Variables
        tree = ET.parse(src_file)
        root = tree.getroot()
        depth_reference = tree_depth * segment_len
        # Data to save
        self.identifiers = []
        self.data = []
        # Go through all description records
        for record in root:
            # Retrieve reference number of all mesh term matching reference depth parameter            
            ref_numbers = record.xpath('./TreeNumberList//TreeNumber[string-length(text()) < ' + str(depth_reference) +']/text()')            
            # Retrieve mesh name - format the list and duplicate name for each reference number entry
            name = [record.xpath('./DescriptorName/String/text()')[0]] * len(ref_numbers)     
            if ref_numbers:
                # Use number as feature name for ease of search later
                self.identifiers.extend(ref_numbers)
                # Use mesh name as value
                self.data.extend(name)

    # Save file(s)
    def save(self, dst_file):
        # Convert to pandas dataframe and save
        df = pd.DataFrame([self.data])
        df.columns = self.identifiers
        df.to_csv(dst_file, index=False)
        pass