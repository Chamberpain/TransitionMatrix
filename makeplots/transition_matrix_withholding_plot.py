def date_difference_plot():
    with open('transition_matrix_datespace_data.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    date,traj_plot,actual_data = zip(*datalist)
    q,residual_mean,residual_std = zip(*actual_data)

    date = np.array(date)
    traj_plot = np.array(traj_plot)
    residual_mean = np.array(residual_mean)
    residual_std = np.array(residual_std)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for n,plot_type in enumerate(np.unique(traj_plot).tolist()):
        mask = traj_plot == plot_type
        date_token = date[mask]
        mean_token = residual_mean[mask]
        std_token = residual_std[mask]
        if n == 0:
            ax.errorbar(date_token,mean_token, yerr=std_token, fmt='o',markersize=12,label=plot_type)
        if n ==1:
            ax.errorbar(date_token,mean_token, yerr=std_token, fmt='o',markersize=6,label=plot_type,zorder=10,alpha =.8)

    ax.set_xlabel('Comparison Matrix Timestep')
    ax.set_ylabel('Mean L2 Norm Difference')
    plt.legend()
    plt.savefig('date_difference_l2')
    plt.close()

    for plot_type in np.unique(traj_plot).tolist():   
        plt.figure()
        plt.title('Difference in Eigen Spectrum for '+plot_type+' Transition Matrices')
        plt.xlabel('Original Eigen Value')
        plt.ylabel('Difference')
        mask = traj_plot==plot_type
        date_token = date[mask]
        q_token = np.array(q)[mask]
        for n,d in enumerate(date_token):
            plot_label = str(d)+' days'
            eigen_spectrum,test_eigen_spectrum = zip(*q_token[n])
            eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
            test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
            mask = eigen_spectrum>0.8
            
            diff = eigen_spectrum[mask]-test_eigen_spectrum[mask]

            plt.plot(eigen_spectrum[mask],diff,label=plot_label)
        plt.legend()
        plt.savefig('date_difference_'+plot_type)
        plt.close()


def data_withholding_plot():

    with open('transition_matrix_withholding_data.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    token,percentage,actual_data = zip(*datalist)
    token = [str(x) for x in token]
    token = np.array(token)
    percentage = np.array(percentage)
    actual_data = np.array(actual_data)
    for t in np.unique(token):
        plt.figure()
        plt.title('Data Withholding for '+str(t))
        mask = token==t
        percentage_token = percentage[mask]
        actual_data_token = actual_data[mask]
        mean_list = []
        std_list = []
        for p in np.unique(percentage_token):
            mask = percentage_token ==p
            plot_label = str(round((1-p)*100))+'% withheld'
            data = actual_data_token[mask]
            eig_list = []
            mean_list_holder = []
            std_list_holder = []
            for n,d in enumerate(data):
                eigs,l2_mean,l2_std = d
                mean_list_holder.append(l2_mean)
                std_list_holder.append(l2_std)
                eigen_spectrum,test_eigen_spectrum = zip(*eigs)
                eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
                test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
                if n == 0:
                    eig_x_coord = eigen_spectrum[eigen_spectrum>0.8]
                    eig_len = len(eig_x_coord)
                diff = eigen_spectrum[-eig_len:]-test_eigen_spectrum[-eig_len:]
                eig_list.append(diff.tolist())
            mean_list.append(np.mean(mean_list_holder))
            std_list.append(np.mean(std_list_holder))            
            plt.plot(eig_x_coord,np.array(eig_list).mean(axis=0),label=plot_label)
        plt.legend()
        plt.xlabel('Eigen Value')
        plt.ylabel('Mean Eigen Value Difference')
        plt.savefig('data_withholding_'+str(t))
        plt.close()
        plt.figure('l2 error')
        plt.errorbar(np.round((1-np.unique(percentage_token))*100),mean_list,yerr=std_list,fmt='o',label=str(t))
    plt.figure('l2 error')
    plt.xlabel('% withheld')
    plt.ylabel('Mean L2 Difference')
    plt.legend()
    plt.savefig('combined_withholding_L2')

def resolution_difference_plot():
    with open('transition_matrix_resolution_comparison.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    inner,outer,traj_file_type, actual_data = zip(*datalist)
    inner = np.array([str(x) for x in inner])
    outer = np.array([str(x) for x in outer])
    traj_file_type = np.array(traj_file_type)
    actual_data = np.array(actual_data)
    for plot_type in np.unique(traj_file_type):
        plt.figure()
        plt.title(plot_type+' Eigen Spectrum')
        mean_list = []
        std_list = []
        mask = traj_file_type == plot_type
        inner_holder = inner[mask]
        outer_holder = outer[mask]
        data_holder = actual_data[mask]
        for n,(inn,out,dat) in enumerate(zip(inner_holder, outer_holder, data_holder)):
            plot_label = str(inn)+','+str(out)
            eigs,l2_mean,l2_std = dat
            mean_list.append(l2_mean)
            std_list.append(std_list)
            eigen_spectrum,test_eigen_spectrum = zip(*eigs)
            eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
            test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
            eig_x_coord = eigen_spectrum[eigen_spectrum>0.8]
            eig_len = len(eig_x_coord)
            diff = eigen_spectrum[-eig_len:]-test_eigen_spectrum[-eig_len:]
            plt.plot(eig_x_coord,diff,label=plot_label)


def resolution_difference_plot():
    with open('transition_matrix_resolution_comparison.pickle','rb') as fp:
        datalist = pickle.load(fp)
    inner,outer,traj_file_type,actual_data = zip(*datalist)
    q,residual_mean,residual_std = zip(*actual_data)
    inner_lat,inner_lon = zip(*inner)
    outer_lat,outer_lon = zip(*outer)
    x_coord = np.array(outer_lon)/np.array(inner_lon)
    y_coord = np.array(outer_lat)/np.array(inner_lat)
    label_list = [str(x[0])+' to '+str(x[1]) for x in zip(inner,outer)]
    colors = cm.rainbow(np.linspace(0, 1, len(x_coord)))
    for n,plot_type in enumerate(np.unique(traj_file_type).tolist()):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mask = np.array(traj_file_type) == plot_type
        mean_token = np.array(residual_mean)[mask]
        for x_coord_token,y_coord_token,mean_token,std_token,label in zip(np.array(x_coord)[mask],np.array(y_coord)[mask],np.array(residual_mean)[mask],np.array(residual_std)[mask],np.array(label_list)[mask]):
            if n == 0:
                ax.errorbar(x_coord_token,y_coord_token,yerr=std_token*25,xerr=std_token*25,marker = 'o',markersize=mean_token*50000,zorder=1/mean_token,label=label)
                plt.title('Argo Resolution Difference Uncertainty')
            if n ==1:
                ax.errorbar(x_coord_token,y_coord_token,yerr=std_token*25,xerr=std_token*25,marker = 'o',markersize=mean_token*50000,zorder=1/mean_token,label=label)
                plt.title('SOSE Resolution Difference Uncertainty')
        plt.xlabel('Ratio of Longitude Resolution')
        plt.ylabel('Ratio of Latitude Resolution')
        plt.legend()

    ax.set_xlabel('Comparison Matrix Timestep')
    ax.set_ylabel('Mean L2 Norm Difference')
    plt.legend()
    plt.savefig('date_difference_l2')
    plt.close()