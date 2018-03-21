from argo_read import traj_file_reader, traj_df,time_parser

root_folder = '/Users/paulchamberlain/Data/Traj/'
df = traj_df(root_folder)
df.to_pickle('global_argo_traj')