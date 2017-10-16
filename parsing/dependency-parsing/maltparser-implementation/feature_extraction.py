# config {stack,buffer,label}
def get_features(config,sent_dict):
    features = []

    # TODO Improve Features

    if len(config[0]) > 0:
        # Top of stack.
        top = config[0][-1]
        
        # Token
        #top_stk_token_feature = 'TOP_STK_TOKEN_'+str(sent_dict['FORM'][top].lower())
        top_stk_postag_feature = 'TOP_STK_POSTAG_'+str(sent_dict['POSTAG'][top].lower())
        top_stk_lemma_feature = 'TOP_STK_LEMMA_'+str(sent_dict['LEMMA'][top].lower())
        
        #features.append(top_stk_token_feature)
        features.append(top_stk_postag_feature)    
        features.append(top_stk_lemma_feature)
    
    if len(config[0]) > 1:
        top2 = config[0][-2]
        
        top2_stk_postag_feature = 'TOP2_STK_POSTAG_'+str(sent_dict['POSTAG'][top2].lower())
        top2_stk_lemma_feature = 'TOP2_STK_LEMMA_'+str(sent_dict['LEMMA'][top2].lower())
        
        features.append(top2_stk_postag_feature)    
        features.append(top2_stk_lemma_feature)
    
    
    if len(config[1]) > 0:
        top_b = config[1][-1]
        
        top_buff_token_feature = 'TOP_BUFF_TOKEN_'+str(sent_dict['LEMMA'][top_b].lower())
        top_buff_postag_feature = 'TOP_BUFF_POSTAG_'+str(sent_dict['POSTAG'][top_b].lower())
    
        features.append(top_buff_token_feature)
        features.append(top_buff_postag_feature)
        
    
    len_sent_feature = 'LEN_SENT_'+str(len(sent_dict['TEXT']))
    features.append(len_sent_feature)
    
    len_form_feature = 'LEN_FORM_'+str(len(sent_dict['FORM']))
    features.append(len_form_feature)
        
    return features
