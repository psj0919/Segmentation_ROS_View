from pyntcloud import PyntCloud
import matplotlib.pyplot as plt





if __name__=='__main__':
    pcd_file = "/storage/sjpark/vehicle_data/Dataset/original_img/img_train/16_105517_220617/sensor_raw_data/lidar/16_105517_220617_01.pcd"
    cloud = PyntCloud.from_file(pcd_file)
    points = cloud.points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points['x'], points['y'], points['z'], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
