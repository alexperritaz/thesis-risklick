# -*- coding: utf-8 -*-

import os
from pathlib import Path

class Accessor:
    
    def __init__(self):
        working_directory = str(Path(*Path(os.getcwd()).parts[:-1]))
        self.project_directory = os.path.expanduser(os.path.dirname(working_directory))
        pass
    
    def create_arborescence(self):
        
        self.directories = {
            # Configuration files
            'config'                : self.project_directory + './configuration/tag_selection/',
            # Raw data
            'all_public_xml'        : self.project_directory + './data/raw/clinical_trials/all_public_xml/',
            'all_trials_xml'        : self.project_directory + './data/raw/clinical_trials/all_trials_xml/',
            # External data
            'external_pubmed'       : self.project_directory + './data/external/pubmed/',
            'external_mesh'         : self.project_directory + './data/external/mesh/',
            # Shared data
            'common_mesh'           : self.project_directory + './data/common/mesh/',
            'common_risks'          : self.project_directory + './data/common/risks/',
            # Analysis directory
            'analysis'              : self.project_directory + './analysis/',
            # Scripts directory
            'script_model'          : self.project_directory + './scripts/models/',
            'script_preprocessing'  : self.project_directory + './scripts/preprocessing/',
            'script_training'       : self.project_directory + './scripts/training/',
            'script_transformers'   : self.project_directory + './scripts/transformers/',
            }
        
        for key, path in self.directories.items():
            Path(path).mkdir(parents=True, exist_ok=True) 
        return self.directories
    
    def files(self, sample_size, configuration):
        files = {
            'src_pubmed'        : self.directories['external_pubmed'] + 'medline.csv',
            'dst_risk_encoded'  : self.directories['common_risks'] + 'risks_encoded.csv',
            'dst_risk_onehot'   : self.directories['common_risks'] + 'risks_onehot.csv',
            'src_mesh'          : self.directories['external_mesh'] + 'MESHdesc2020.xml',
            'all_mesh'          : self.directories['common_mesh'] + 'mesh_all.csv',
            'src_config'        : self.directories['config'] + '%s.csv' % (configuration),
            'features_extracted': self.directories_config['extracted'] + 'features.csv',
            'labels_sampled'    : self.directories_config['sampled'] + 'labels_encoded.csv',
            'converted'         : self.directories_config['sampled'] + 'converted.csv',
            'cleaned'           : self.directories_config['sampled'] + 'cleaned.csv',
            'transformed'       : self.directories_config['sampled'] + 'transformed.csv',
            'encoded'           : self.directories_config['sampled'] + 'encoded.csv',
            'encoded_offcut'    : self.directories_config['sampled'] + 'encoded_offcut.csv',
            'embedded'          : self.directories_config['sampled'] + 'embedded',
            'encoded_analysis'  : self.directories_config['sampled_analysis'] + 'encoded_info.csv',
            'embedded_logs'     : self.directories_config['sampled_analysis'] + 'embedded_logs.csv',
            }
        return files
    
    def create_specific(self, sample_size, configuration):
        
        self.directories_config = {
            'extracted_xml' : self.project_directory + './data/interim/%s/extracted/xml/' % (configuration),
            'extracted'     : self.project_directory + './data/interim/%s/extracted/' % (configuration),
            'sampled'       : self.project_directory + './data/interim/%s/sampled_%s_per_class/' % (configuration, sample_size),
            'sampled_xml'   : self.project_directory + './data/interim/%s/sampled_%s_per_class/xml/' % (configuration, sample_size),
            'sampled_analysis' : self.project_directory + './data/interim/%s/sampled_%s_per_class/analysis/' % (configuration, sample_size),
            }
        
        for key, path in self.directories_config.items():
            Path(path).mkdir(parents=True, exist_ok=True)         
        return self.directories_config
    
    def get_ordinal_filter():
        filter_list = ['/clinical_study/eligibility/maximum_age',
                       '/clinical_study/eligibility/minimum_age', 
                       '/clinical_study/enrollment',
                       # '/clinical_study/phase',
                       # '/clinical_study/study_design_info/masking', 
                       'duration']
        return filter_list
    
    def get_advanced_filter():
        # Eligibility selected features
        eligibility         = ['/clinical_study/eligibility/criteria/textblock',
                               '/clinical_study/eligibility/gender',
                               '/clinical_study/eligibility/healthy_volunteers',
                               '/clinical_study/eligibility/maximum_age',
                               '/clinical_study/eligibility/minimum_age']
        # Intervention selected features
        intervention        = ['/clinical_study/intervention/intervention_name',
                               '/clinical_study/intervention/intervention_type']
        # Location selected features
        location            = ['/clinical_study/location/facility/name']
        # Primary outcome selected features
        primary_outcome     = ['/clinical_study/primary_outcome',	
                               # '/clinical_study/primary_outcome/description',
                               '/clinical_study/primary_outcome/measure']
        # Sponsors selected features
        sponsors            = ['/clinical_study/sponsors/lead_sponsor/agency']
        # Study design info selected features
        study_design_info   = ['/clinical_study/study_design_info', 
                               '/clinical_study/study_design_info/allocation', 
                               '/clinical_study/study_design_info/intervention_model', 
                               # '/clinical_study/study_design_info/intervention_model_description', 
                               '/clinical_study/study_design_info/masking', 
                               # '/clinical_study/study_design_info/masking_description', 
                               '/clinical_study/study_design_info/primary_purpose']
        # All selected features
        filter_list = {'/clinical_study/eligibility/'       : eligibility,
                       '/clinical_study/intervention/'      : intervention,
                       '/clinical_study/location/'          : location,
                       '/clinical_study/primary_outcome/'   : primary_outcome,
                       '/clinical_study/sponsors/'          : sponsors,
                       '/clinical_study/study_design_info/' : study_design_info}
        return filter_list
    
    