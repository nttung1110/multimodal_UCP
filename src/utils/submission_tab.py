import os
import os as osp

def prepare_output_all_files(path_out_folder, 
                            path_out_idx_file, 
                            all_result):
    
    # all_result is a list of predicted result for each modality [video_dict, audio_dict, text_dict]

    h_so_line = '\t'.join(['file_id', 'is_processed', 'message', 'file_path'])
    all_line_index = [h_so_line] # for system_out.index.tab file

    for result_dict in all_result:
        for each_file in result_dict:
            header_line = '\t'.join(['file_id', 'timestamp', 'llr'])

            all_lines = [header_line]
            cp_loc, cp_llr = result_dict[each_file]['cp_pos'],\
                                result_dict[each_file]['llr']
            
            if len(cp_loc) == 0:
                message = 'no output'
            else:
                message = ''
            
            for each_cp, each_llr in zip(cp_loc, cp_llr):
                content_line = [each_file, str(each_cp-1), '{:0.2f}'.format(each_llr)]
                content_line = '\t'.join(content_line)
                all_lines.append(content_line)

            # write single file
            result = '\n'.join(all_lines)

            path_out_file = osp.join(path_out_folder, each_file+'.tab')
            with open(path_out_file, 'w') as fp:
                fp.write(result)

            # also construct index for this file
            line_idx = [each_file, 'True', message, each_file+'.tab']
            line_idx = '\t'.join(line_idx)
            all_line_index.append(line_idx)

    all_line_index = '\n'.join(all_line_index)
    with open(path_out_idx_file, 'w') as fp:
        fp.write(all_line_index)
