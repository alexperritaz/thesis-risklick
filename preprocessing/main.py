# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:09:00 2020

@author: Alex
"""
# Data Accessor
from accessor import Accessor
# Data preparation - Common
from structure import Structure
from risk_assessment import RiskAssessment
from meshify import MeSH
# Data preparation - General
from downsample import DownSample
from extract import Extract
from convert import Convert
# Data formatting / cleaning
from clean import Clean
from transform import Transform
# Data conversion for machine interpretation
from encode import Encode
from embed import Embed
from embed_advanced import EmbedBERT

from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, TransformerDocumentEmbeddings, DocumentPoolEmbeddings


class Main(object):
    
    def __init__(self):
        
        pass

    
    def run(self):
        class_sample_size = 20000
        # class_sample_size = 'all'
        configuration = 'tags_few'
        study_types = ['interventional']
        segment_length = 4
        depth = 3
        whole_document = False
        self.define_paths(class_sample_size, configuration)
        # self.prepare_common(study_types, segment_length, depth) # Run once
        # self.prepare_data(class_sample_size, whole_document, study_types)
        # self.format_data(segment_length, depth)
        self.translate_data()
        
        pass

    def define_paths(self, sample_size, configuration):
        accessor = Accessor()
        self.directories = accessor.create_arborescence()
        self.directories_config = accessor.create_specific(sample_size, configuration)
        self.files = accessor.files(sample_size, configuration)
        pass
    
    def prepare_common(self, study_types, segment_length, depth):
        self.restructure(self.directories['all_public_xml'], self.directories['all_trials_xml'])
        self.assess_risks(self.directories['all_trials_xml'], self.files['src_pubmed'], self.files['dst_risk_encoded'], self.files['dst_risk_onehot'], study_types)
        self.meshify(self.directories['common_mesh'], self.files['src_mesh'], self.files['all_mesh'], segment_length, depth)
        pass
    
    def prepare_data(self, sample_size, whole_document, study_types):
        # self.extract(self.directories['all_trials_xml'], self.directories_config['extracted_xml'], self.files['src_config'], self.files['features_extracted'], whole_document, study_types)
        self.downsize(self.directories_config['extracted_xml'], self.directories_config['sampled_xml'], self.files['dst_risk_encoded'], self.files['labels_sampled'], sample_size)
        # self.convert(self.directories_config['sampled_xml'], self.files['features_extracted'], self.files['converted'], whole_document)
        pass
    
    def format_data(self, segment_length, depth):        
        self.clean(self.files['converted'], self.files['cleaned'])
        self.transform(self.files['cleaned'], self.files['all_mesh'], self.directories['common_mesh'] + 'mesh_depth_%s.csv' % (str(depth)) , self.files['transformed'], segment_length, depth)
        pass
    
    def translate_data(self):
        
        # self.encode(self.files['transformed'], self.files['encoded'], self.files['encoded_offcut'], self.files['encoded_analysis'])
        self.embed(self.files['encoded_offcut'], self.files['embedded'], self.files['embedded_logs'])
        
        pass
    
    def restructure(self, src_directory, dst_directory):
        #
        structure = Structure()
        structure.from_subdirectories(src_directory, dst_directory)
        pass
    
    def assess_risks(self, src_directory, src_file, dst_file_encoded, dst_file_onehot, study_types):
        #
        status_ongoing = ['not yet recruiting','recruiting','active, not recruiting','enrolling by invitation','available','temporarily not available']
        status_terminated = ['terminated','suspended','withdrawn','unknown status','no longer available','withheld']
        ct_status = [status_ongoing, status_terminated]
        #        
        risk_assessment = RiskAssessment(src_directory, src_file, ct_status, study_types)
        risk_assessment.process()
        risk_assessment.save(dst_file_encoded)
        risk_assessment.to_onehot()
        risk_assessment.save(dst_file_onehot)
        pass
    
    def meshify(self, dst_directory, src_file, dst_file, segment_length, depth):
        #
        mesh = MeSH()
        mesh.extract_all(src_file)
        mesh.save(dst_file)
        #
        mesh.extract_level(src_file, depth, segment_length)
        filename = 'mesh_depth_%s.csv' % (str(depth))
        mesh.save(dst_directory + filename)
        pass
    
    def extract(self, src_directory, dst_directory, src_file, dst_file, whole_document, study_types):
        #
        extract = Extract(src_directory, dst_directory, src_file, whole_document, study_types)
        extract.process()
        extract.save(dst_file)
        pass

    def downsize(self, src_directory, dst_directory, src_file, dst_file, sample_size):
        #
        downsample = DownSample(src_directory, dst_directory, src_file, sample_size)
        downsample.downsize(dst_file)
        pass
    
    def convert(self, src_directory, src_file, dst_file, whole_document):
        #
        convert = Convert(src_directory, src_file, whole_document)
        convert.process()
        convert.save(dst_file)
        pass
    
    def clean(self, src_file, dst_file):
        clean = Clean(src_file)
        clean.filter_out(Accessor.get_advanced_filter())
        clean.str_to_int()
        clean.replace_nan()
        clean.remove_default()
        # clean.data_imputation()
        clean.age_to_days()
        clean.date_to_timestamp()
        clean.save(dst_file)
        pass
    
    def transform(self, src_file, src_file_mesh_ref, src_file_mesh_lvl, dst_file, segment_length, depth):
        transform = Transform(src_file)
        transform.define_columns()
        transform.convert_age()
        transform.convert_date()
        transform.bin_data()
        transform.meshify_data(src_file_mesh_ref, src_file_mesh_lvl, segment_length, depth)
        transform.save(dst_file)
        pass
    
    def encode(self, src_file, dst_file, dst_file_offcut, dst_file_analysis):
        threshold_label_encoding = 30
        threshold_multihot_encoding = 3000
        #
        encode = Encode(src_file)
        encode.encode_features(threshold_label_encoding, threshold_multihot_encoding, Accessor.get_ordinal_filter())
        encode.save(dst_file, dst_file_offcut, dst_file_analysis)
        pass
    
    def embed(self, src_file, dst_file, dst_log):
        glove_embedding = WordEmbeddings('glove')
        # flair_embedding_forward = FlairEmbeddings('news-forward')
        # flair_embedding_backward = FlairEmbeddings('news-backward')
        
        embedding = DocumentPoolEmbeddings([glove_embedding])
        
        # Out of mememory
        # embedding = TransformerDocumentEmbeddings('bert-base-uncased', layers='-1,-2,-3,-4', use_scalar_mix=True) 
        
        
        # embedding =  TransformerWordEmbeddings('bert-base-uncased', layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean", allow_long_sentences=True) 
        # embedding =  TransformerWordEmbeddings('bert-base-uncased', layers="all", use_scalar_mix=True, pooling_operation="first", allow_long_sentences=True) 
        #        
        
        embedding_name = '_bert_document_mean'
        dst_file = dst_file + embedding_name + '.csv'
        
        embed = Embed(src_file, dst_file, dst_log)
        embed.define(embedding)
        embed.process()
        embed.save()
        
        # embed = EmbedBERT(src_file, dst_file, dst_log)
        # embed.define(embedding)
        # embed.process()
        # embed.save()
        pass
    
if __name__ == "__main__":
    main = Main()
    main.run()