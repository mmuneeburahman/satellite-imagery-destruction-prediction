import gradio as gr

def predict(pre_file, post_file):
    pass

iface = gr.Interface(fn=predict, 
                     inputs=[
                         gr.File(label="Pre destruction tiff file"),
                         gr.File(label="Post destruction tiff file")
                         ],
                     outputs=[
                        gr.File(label="Localization mask"),
                        gr.File(label="Localization overlay"),
                        gr.File(label="Destruction mask"),
                        gr.File(label="Destruction mask overlay")
                        ],
                     title="TIFF Image Processing",
                     description="Convert, invert, rotate, and flip TIFF images"
    )
iface.launch(server_name="0.0.0.0", server_port=3092)