# -*- coding: utf-8 -*-
import re 
import lxml.etree as ET
from utils import walk_files, progress_bar, read_csv, write_csv


class Extract:
    
    def __init__(self, src_directory, dst_directory, src_file, whole_document, study_type):
        self.src_directory = src_directory
        self.dst_directory = dst_directory
        self.tags = read_csv(src_file)
        self.study_type = study_type
        self.filter_study = self.filter_type if self.study_type else self.filter_none
        self.perform_action = self.remove_nodes if whole_document else  self.filter_nodes
        self.headers = ['nct_id']
        pass
    
    # Proceed to data extraction
    def process(self):
        self.tags = [x.lower() for x in self.tags]
        filepaths = walk_files(self.src_directory)
        for filepath in progress_bar(filepaths, "Processing files : "):
            self.process_file(filepath)
        self.headers[1:] = sorted(self.headers[1:])
        pass
    
    # Process steps
    def process_file(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        if self.filter_study(tree):
            self.filter_comments(tree)
            self.perform_action(root)
            self.clip_features(tree, root)
            self.xml_file(filepath, root)
        pass
    
    # Filter out non-interventionnal trials
    def filter_type(self, tree):
        return tree.xpath('//study_type/text()')[0].lower() in self.study_type
    
    # Filter - none
    def filter_none(self, tree):
        return True

    # Filter out comments
    def filter_comments(self, tree):
        for comment in tree.xpath('//comment()'):
            comment.getparent().remove(comment)
        pass
    
    # Filter out non-specified nodes
    def filter_nodes(self, root):
        for node in root:
            if node.tag.lower() not in self.tags:
                root.remove(node)
        pass
    
    # Remove nodes containing result information
    def remove_nodes(self, root):        
        for node in root:
            if node.tag.lower() in self.tags:
                root.remove(node)
        pass
        
    # Retrieve all selected nodes (including nested ones)
    def clip_features(self, tree, root):
        for node in root.iter():
            featurename = self.format_path(tree, node)
            if featurename not in self.headers:
                self.headers.append(featurename.lower())
        pass
    
    # Remove '[#]' to keep general features
    def format_path(self, tree, node):
        path = tree.getpath(node)
        path = re.sub("[\(\[].*?[\)\]]", "", path)
        return path.lower()
    
    # Create new XML files with extracted info
    def xml_file(self, filepath, root):
        newTree = ET.ElementTree(root)
        filename = filepath.split('/')[-1]
        newTree.write(self.dst_directory + filename, pretty_print=True)
        pass
    
    # Save extracted features
    def save(self, dst_file):
        write_csv(dst_file, self.headers, False)