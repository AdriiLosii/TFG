#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <arv.h>
#include "std_msgs/msg/string.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sstream>

#include <image_transport/image_transport.h>

#define ARV_PIXEL_FORMAT_BYTE_PER_PIXEL(pixel_format) ((((pixel_format) >> 16) & 0xff) >> 3)

int frameCount = 0;
double fps = 0.0;
auto startTime = std::chrono::high_resolution_clock::now();
struct global_s
{
	gboolean 								bCancel;
	std::shared_ptr<rclcpp::Node>			*node;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr textPublisher;
	image_transport::Publisher 				*publisher;
	gint 									width, height;

    int         							xRoi;
	int         							yRoi;
	int         							widthRoi;
	int										widthRoiMin;
	int										widthRoiMax;
	int         							heightRoi;
	int										heightRoiMin;
	int										heightRoiMax;
	int										exposureTime;
	int										acquisitionMode;
	ArvAcquisitionMode						arvAcquisitionMode;
	double									frameRate;

	std::string								ipAddress;

	const char                             *pszPixelformat;
	unsigned								nBytesPixel;
	ArvCamera 							   *pCamera;
	ArvDevice 							   *pDevice;
	ArvStream                              *pStream;

} global;

typedef struct {
	GMainLoop *main_loop;
	int buffer_count;
} ApplicationData;

static gboolean cancel = FALSE;

static void
set_cancel (int signal)
{
	cancel = TRUE;
}

static void
new_buffer_cb (ArvStream *stream, ApplicationData *data)
{
	
	static uint64_t  cm = 0L;	// Camera time prev
	uint64_t  		 cn = 0L;	// Camera time now
	static uint32_t	 iFrame = 0;	// Frame counter.

	ArvBuffer *buffer;

	buffer = arv_stream_try_pop_buffer (stream);
	if (buffer != NULL)
	{
		if (arv_buffer_get_status (buffer) == ARV_BUFFER_STATUS_SUCCESS)
		{
			sensor_msgs::msg::Image imageMessage;
			//std_msgs::msg::String textMessage;
			data->buffer_count++;

			size_t buffer_size;
			const uint8_t * buffer_data = static_cast<const uint8_t *>(arv_buffer_get_data(buffer, &buffer_size));
			std::vector<uint8_t> this_data(buffer_size);
			memcpy(&this_data[0], buffer_data, buffer_size);

			// Camera/ROS Timestamp coordination.
			//cn = (uint64_t)arv_buffer_get_timestamp (buffer);		// Camera now
			//rn = ros::Time::now().toNSec();						// ROS now

			imageMessage.header.frame_id = "camera";
			imageMessage.width = global.widthRoi;
			imageMessage.height = global.heightRoi;
			imageMessage.encoding = "bayer_rggb8";
			imageMessage.step = imageMessage.width * global.nBytesPixel;
			imageMessage.data = this_data;
			global.publisher->publish(imageMessage);

			frameCount++;
			auto currentTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsedSeconds = currentTime - startTime;

			if (elapsedSeconds.count() >= 1.0)
			{
				fps = frameCount / elapsedSeconds.count(); // FPS = frames / seconds
				std::cout << "FPS: " << fps << std::endl;

				// Reset counters
				frameCount = 0;
				startTime = currentTime;
        	}
		}
			
		arv_stream_push_buffer (stream, buffer);
		iFrame++;
	}
}

static gboolean
periodic_task_cb (void *abstract_data)
{
	ApplicationData *data = static_cast<ApplicationData*>(abstract_data);

	data->buffer_count = 0;

	if (cancel) {
		g_main_loop_quit (data->main_loop);
		return FALSE;
	}

	rclcpp::spin_some(*global.node);

	return TRUE;
}

static void
control_lost_cb (ArvGvDevice *gv_device)
{
	/* Control of the device is lost. Display a message and force application exit */
	printf ("Control lost\n");

	// arv_camera_stop_acquisition(global.pCamera, NULL);
	// arv_stream_set_emit_signals (global.pStream, FALSE);
	// g_object_unref(global.pStream);
	// g_object_unref(global.pCamera);

	// global.pCamera = arv_camera_new (global.ipAddress.c_str(), NULL);
	// global.pStream = arv_camera_create_stream (global.pCamera, NULL, NULL, NULL);
	// arv_camera_start_acquisition (global.pCamera, NULL);
	// arv_stream_set_emit_signals (global.pStream, TRUE);

	cancel = TRUE;
}

int main (int argc, char **argv)
{	
	std::cout << "Initializing ROS2.." << std::endl;
	// ROS2 initialization
	rclcpp::init(argc, argv);
	// Node definition
	rclcpp::NodeOptions options;
	global.node = new std::shared_ptr<rclcpp::Node>(std::make_shared<rclcpp::Node>("publisher", options));
	// Publisher definition
	image_transport::ImageTransport it(*global.node);
	global.publisher = new image_transport::Publisher(it.advertise("image", 1));

	std::cout << "Loading parameters.." << std::endl;
	(*global.node)->declare_parameter("ip_address", rclcpp::PARAMETER_STRING);
	(*global.node)->declare_parameter("frame_rate", rclcpp::PARAMETER_DOUBLE);
	(*global.node)->declare_parameter("exposure_time", rclcpp::PARAMETER_INTEGER);
	(*global.node)->declare_parameter("acquisition_mode", rclcpp::PARAMETER_INTEGER);
	//std::string ip_address = (*global.node)->get_parameter("ip_address").as_string();
	global.ipAddress = (*global.node)->get_parameter("ip_address").as_string();
	global.frameRate = (*global.node)->get_parameter("frame_rate").as_double();
	global.exposureTime = (*global.node)->get_parameter("exposure_time").as_int();
	global.acquisitionMode = (*global.node)->get_parameter("acquisition_mode").as_int();

	switch (global.acquisitionMode)
	{
	case 0:
		global.arvAcquisitionMode = ARV_ACQUISITION_MODE_CONTINUOUS;
		break;
	case 1:
		global.arvAcquisitionMode = ARV_ACQUISITION_MODE_SINGLE_FRAME;
		std::cout << "This acquisition mode is not currently supported.." << std::endl;
		rclcpp::shutdown();
		return 0;
		break;
	case 2:
		global.arvAcquisitionMode = ARV_ACQUISITION_MODE_MULTI_FRAME;
		std::cout << "This acquisition mode is not currently supported.." << std::endl;
		rclcpp::shutdown();
		return 0;
		break;
	
	default:
		break;
	}

	ApplicationData data;
	ArvCamera *camera;
	ArvStream *stream;
	ArvGcNode *pGcNode;
	GError *error = NULL;
	int i;

	data.buffer_count = 0;

	//while(rclcpp::ok()){
	cancel = FALSE;
	/* Instantiation of the first available camera */
	std::cout << "Connecting to camera.." << std::endl;
	global.pCamera = arv_camera_new (global.ipAddress.c_str(), &error);

	if (ARV_IS_CAMERA (global.pCamera)) {
		void (*old_sigint_handler)(int);
		gint payload;

		std::cout << "Configuring camera.." << std::endl;

		// Acquiring values from camera
		global.pDevice = arv_camera_get_device(global.pCamera);
		arv_camera_get_width_bounds(global.pCamera, &global.widthRoiMin, &global.widthRoiMax, &error);
		arv_camera_get_height_bounds(global.pCamera, &global.heightRoiMin, &global.heightRoiMax, &error);
		arv_camera_get_region(global.pCamera, &global.xRoi, &global.yRoi, &global.widthRoi, &global.heightRoi, &error);
		global.pszPixelformat = g_string_ascii_down(g_string_new(arv_device_get_string_feature_value(global.pDevice, "PixelFormat", &error)))->str;
		global.nBytesPixel = ARV_PIXEL_FORMAT_BYTE_PER_PIXEL(arv_device_get_integer_feature_value(global.pDevice, "PixelFormat", &error));
		
		arv_camera_set_frame_rate(global.pCamera, global.frameRate, NULL);
		double actualFramerate = arv_camera_get_frame_rate (global.pCamera, &error);
		std::cout << "Framerate: " << actualFramerate << std::endl;

		arv_camera_set_exposure_time(global.pCamera, global.exposureTime, NULL);
		double exposureTime = arv_camera_get_exposure_time(global.pCamera, &error);
		std::cout << "Exposure time: " << exposureTime << std::endl;

		arv_camera_set_acquisition_mode(global.pCamera, global.arvAcquisitionMode, NULL);
		ArvAcquisitionMode currentMode = arv_camera_get_acquisition_mode(global.pCamera, &error);
		std::cout << "Acquisition mode: " << currentMode << std::endl;
		
		/* retrieve image payload (number of bytes per image) */
		payload = arv_camera_get_payload (global.pCamera, NULL);
		std::cout <<  "Payload: " << payload << std::endl;

		/* Create a new stream object */
		global.pStream = arv_camera_create_stream (global.pCamera, NULL, NULL, &error);
		auto startTime = std::chrono::high_resolution_clock::now();
		if (ARV_IS_STREAM (global.pStream)) {
			std::cout << "Running!" << std::endl;
			/* Push 50 buffer in the stream input buffer queue */
			for (i = 0; i < 50; i++)
				arv_stream_push_buffer (global.pStream, arv_buffer_new (payload, NULL));

			/* Start the video stream */
			arv_camera_start_acquisition (global.pCamera, NULL);

			/* Connect the new-buffer signal */
			g_signal_connect (global.pStream, "new-buffer", G_CALLBACK (new_buffer_cb), &data);
			/* And enable emission of this signal (it's disabled by default for performance reason) */
			arv_stream_set_emit_signals (global.pStream, TRUE);

			/* Connect the control-lost signal */
			g_signal_connect (arv_camera_get_device (global.pCamera), "control-lost",
					  G_CALLBACK (control_lost_cb), NULL);

			/* Install the callback for frame rate display */
			g_timeout_add_seconds (1, periodic_task_cb, &data);
			//g_timeout_add (5, periodic_task_cb, &data);

			/* Create a new glib main loop */
			data.main_loop = g_main_loop_new (NULL, FALSE);

			old_sigint_handler = signal (SIGINT, set_cancel);

			/* Run the main loop */
			g_main_loop_run (data.main_loop);

			signal (SIGINT, old_sigint_handler);

			g_main_loop_unref (data.main_loop);

			/* Stop the video stream */
			arv_camera_stop_acquisition (global.pCamera, NULL);

			/* Signal must be inhibited to avoid stream thread running after the last unref */
			arv_stream_set_emit_signals (global.pStream, FALSE);

			g_object_unref (global.pStream);
		} else {
			printf ("Can't create stream thread%s%s\n",
				error != NULL ? ": " : "",
				error != NULL ? error->message : "");

			g_clear_error (&error);
		}

		g_object_unref (global.pCamera);
	} else {
		printf ("No camera found%s%s\n",
			error != NULL ? ": " : "",
			error != NULL ? error->message : "");
		g_clear_error (&error);
	}

	//} // while(rclcpp::ok)
	
	delete global.publisher;
	delete global.node;

	rclcpp::shutdown();
	return 0;
}
