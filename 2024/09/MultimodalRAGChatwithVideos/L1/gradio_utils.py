import gradio as gr
import io
import sys
import time
import dataclasses
from pathlib import Path
import os
from enum import auto, Enum
from typing import List, Tuple, Any
from utils import prediction_guard_llava_conv
import lancedb
from utils import load_json_file
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from PIL import Image
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils import prediction_guard_llava_conv, encode_image, Conversation, lvlm_inference_with_conversation

server_error_msg="**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

# function to split video at a timestamp
def split_video(video_path, timestamp_in_ms, output_video_path: str = "./shared_data/splitted_videos", output_video_name: str="video_tmp.mp4", play_before_sec: int=3, play_after_sec: int=3):
    timestamp_in_sec = int(timestamp_in_ms / 1000)
    # create output_video_name folder if not exist:
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    output_video = os.path.join(output_video_path, output_video_name)
    with VideoFileClip(video_path) as video:
        duration = video.duration
        start_time = max(timestamp_in_sec - play_before_sec, 0)
        end_time = min(timestamp_in_sec + play_after_sec, duration)
        new = video.subclip(start_time, end_time)
        new.write_videofile(output_video, audio_codec='aac')
    return output_video


prompt_template = """The transcript associated with the image is '{transcript}'. {user_query}"""

# define default rag_chain
def get_default_rag_chain():
    # declare host file
    LANCEDB_HOST_FILE = "./shared_data/.lancedb"
    # declare table name
    TBL_NAME = "demo_tbl"
    
    # initialize vectorstore
    db = lancedb.connect(LANCEDB_HOST_FILE)

    # initialize an BridgeTower embedder 
    embedder = BridgeTowerEmbeddings()

    ## Creating a LanceDB vector store 
    vectorstore = MultimodalLanceDB(uri=LANCEDB_HOST_FILE, embedding=embedder, table_name=TBL_NAME)
    ### creating a retriever for the vector store
    retriever_module = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})

    # initialize a client as PredictionGuardClien
    client = PredictionGuardClient()
    # initialize LVLM with the given client
    lvlm_inference_module = LVLM(client=client)
    
    def prompt_processing(input):
        # get the retrieved results and user's query
        retrieved_results, user_query = input['retrieved_results'], input['user_query']
        # get the first retrieved result by default
        retrieved_result = retrieved_results[0]
        # prompt_template = """The transcript associated with the image is '{transcript}'. {user_query}"""
        
        # get all metadata of the retrieved video segment
        metadata_retrieved_video_segment = retrieved_result.metadata['metadata']
    
        # get the frame and the corresponding transcript, path to extracted frame, path to whole video, and time stamp of the retrieved video segment.
        transcript = metadata_retrieved_video_segment['transcript']
        frame_path = metadata_retrieved_video_segment['extracted_frame_path']
        return {
            'prompt': prompt_template.format(transcript=transcript, user_query=user_query),
            'image' : frame_path,
            'metadata' : metadata_retrieved_video_segment,
        }
    # initialize prompt processing module as a Langchain RunnableLambda of function prompt_processing
    prompt_processing_module = RunnableLambda(prompt_processing)

    # the output of this new chain will be a dictionary
    mm_rag_chain_with_retrieved_image = (
                    RunnableParallel({"retrieved_results": retriever_module , 
                                      "user_query": RunnablePassthrough()}) 
                    | prompt_processing_module
                    | RunnableParallel({'final_text_output': lvlm_inference_module, 
                                        'input_to_lvlm' : RunnablePassthrough()})
                   )
    return mm_rag_chain_with_retrieved_image
    
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    
@dataclasses.dataclass
class GradioInstance:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"
    sep2: str = None
    version: str = "Unknown"
    path_to_img: str = None
    video_title: str = None
    path_to_video: str = None
    caption: str = None
    mm_rag_chain: Any = None
    
    skip_next: bool = False

    def _template_caption(self):
        out = ""
        if self.caption is not None:
            out = f"The caption associated with the image is '{self.caption}'. "
        return out
    
    def get_prompt_for_rag(self):
        messages = self.messages
        assert len(messages) == 2, "length of current conversation should be 2"
        assert messages[1][1] is None, "the first response message of current conversation should be None"
        ret = messages[0][1]           
        return ret
        
    def get_conversation_for_lvlm(self):
        pg_conv = prediction_guard_llava_conv.copy()
        image_path = self.path_to_img
        b64_img = encode_image(image_path)
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if msg is None:
                break
            if i == 0:
                pg_conv.append_message(prediction_guard_llava_conv.roles[0], [msg, b64_img])
            elif i == len(self.messages[self.offset:]) - 2:
                pg_conv.append_message(role, [prompt_template.format(transcript=self.caption, user_query=msg)])
            else:
                pg_conv.append_message(role, [msg])
        return pg_conv
                
    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        if self.path_to_img is not None:
            path_to_image = self.path_to_img
            images.append(path_to_image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return GradioInstance(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            mm_rag_chain=self.mm_rag_chain,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "path_to_img": self.path_to_img,
            "video_title" : self.video_title,
            "path_to_video": self.path_to_video,
            "caption" : self.caption,
        }
    def get_path_to_subvideos(self):
        if self.video_title is not None and self.path_to_img is not None:
            info = video_helper_map[self.video_title]
            path = info['path']
            prefix = info['prefix']
            vid_index = self.path_to_img.split('/')[-1]
            vid_index = vid_index.split('_')[-1]
            vid_index = vid_index.replace('.jpg', '')
            ret = f"{prefix}{vid_index}.mp4"
            ret = os.path.join(path, ret)
            return ret
        elif self.path_to_video is not None:
            return self.path_to_video
        return None

def get_gradio_instance(mm_rag_chain=None):
    if mm_rag_chain is None:
        mm_rag_chain = get_default_rag_chain()
        
    instance = GradioInstance(
        system="",
        roles=prediction_guard_llava_conv.roles,
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n",
        path_to_img=None,
        video_title=None,
        caption=None,
        mm_rag_chain=mm_rag_chain,
    )
    return instance

gr.set_static_paths(paths=["./assets/"])
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#eff6ff", c500="#0054ae", c600="#00377c", c700="#00377c", c800="#1e40af", c900="#1e3a8a", c950="#0a0c2b"),
    secondary_hue=gr.themes.Color(
        c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#eff6ff", c500="#0054ae", c600="#0054ae", c700="#0054ae", c800="#1e40af", c900="#1e3a8a", c950="#1d3660"),
).set(
    body_background_fill_dark='*primary_950',
    body_text_color_dark='*neutral_300',
    border_color_accent='*primary_700',
    border_color_accent_dark='*neutral_800',
    block_background_fill_dark='*primary_950',
    block_border_width='2px',
    block_border_width_dark='2px',
    button_primary_background_fill_dark='*primary_500',
    button_primary_border_color_dark='*primary_500'
)

css='''
    @font-face {
        font-family: IntelOne;
        src: url("/file=./assets/intelone-bodytext-font-family-regular.ttf");
    }
    .gradio-container {background-color: #0a0c2b}
    table {
      border-collapse: collapse;
      border: none;
    }
'''

##     <td style="border-bottom:0"><img src="file/assets/DCAI_logo.png" height="300" width="300"></td>

# html_title = '''
# <table style="bordercolor=#0a0c2b; border=0">
# <tr style="height:150px; border:0">
#     <td style="border:0"><img src="/file=../assets/intel-labs.png" height="100" width="100"></td>
#     <td style="vertical-align:bottom; border:0"> 
#     <p style="font-size:xx-large;font-family:IntelOne, Georgia, sans-serif;color: white;">
#      Multimodal RAG:
#      <br>
#      Chat with Videos
#     </p>
#     </td>
#     <td style="border:0"><img src="/file=../assets/gaudi.png" width="100" height="100"></td>
    
#     <td style="border:0"><img src="/file=../assets/IDC7.png" width="300" height="350"></td>
#     <td style="border:0"><img src="/file=../assets/prediction_guard3.png" width="120" height="120"></td>
# </tr>
# </table>

# '''

html_title = '''
<table style="bordercolor=#0a0c2b; border=0">
<tr style="height:150px; border:0">
    <td style="border:0"><img src="/file=./assets/header.png"></td>
</tr>
</table>

'''

#<td style="border:0"><img src="/file=../assets/xeon.png" width="100" height="100"></td>
dropdown_list = [
    "What is the name of one of the astronauts?", 
    "An astronaut's spacewalk",
    "What does the astronaut say?", 
    
]

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

def clear_history(state, request: gr.Request):
    state = get_gradio_instance(state.mm_rag_chain)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 1

def add_text(state, text, request: gr.Request):
    if len(text) <= 0 :
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 1

    text = text[:1536]  # Hard cut-off

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 1
    
def http_bot(
    state, request: gr.Request
):
    start_tstamp = time.time()

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        path_to_sub_videos = state.get_path_to_subvideos()
        yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (no_change_btn,) * 1
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = get_gradio_instance(state.mm_rag_chain)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    all_images = state.get_images(return_pil=False)

    # Make requests
    is_very_first_query = True
    if len(all_images) == 0:
        # first query need to do RAG
        # Construct prompt
        prompt_or_conversation = state.get_prompt_for_rag()
    else:
        # subsequence queries, no need to do Retrieval
        is_very_first_query = False
        prompt_or_conversation = state.get_conversation_for_lvlm() 
    
    if is_very_first_query:
        executor = state.mm_rag_chain
    else:
        executor = lvlm_inference_with_conversation
    
    state.messages[-1][-1] = "▌"
    path_to_sub_videos = state.get_path_to_subvideos()
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (disable_btn,) * 1

    try:
        if is_very_first_query:            
            # get response by invoke executor chain
            response = executor.invoke(prompt_or_conversation)
            message = response['final_text_output']
            if 'metadata' in response['input_to_lvlm']:
                metadata = response['input_to_lvlm']['metadata']
                if (state.path_to_img is None 
                    and 'input_to_lvlm' in response 
                    and 'image' in response['input_to_lvlm']
                   ):
                        state.path_to_img = response['input_to_lvlm']['image']
                       
                if state.path_to_video is None and 'video_path' in metadata:
                        video_path = metadata['video_path']
                        mid_time_ms = metadata['mid_time_ms']
                        splited_video_path = split_video(video_path, mid_time_ms)
                        state.path_to_video = splited_video_path
                    
                if state.caption is None and 'transcript' in metadata:
                    state.caption = metadata['transcript']
            else:
                raise ValueError("Response's format is changed")
        else:
            # get the response message by directly call PredictionGuardAPI
            message = executor(prompt_or_conversation)
            
    except Exception as e:
        print(e)
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), None) + (
            enable_btn,
        )
        return

    state.messages[-1][-1] = message
    path_to_sub_videos = state.get_path_to_subvideos()
    # path_to_image = state.path_to_img
    # caption = state.caption
    # # print(path_to_sub_videos)
    # # print(path_to_image)
    # # print('caption: ', caption)
    yield (state, state.to_gradio_chatbot(), path_to_sub_videos) + (enable_btn,) * 1

    finish_tstamp = time.time()
    return
    
def get_demo(rag_chain=None):
    if rag_chain is None:
        rag_chain = get_default_rag_chain()
        
    with gr.Blocks(theme=theme, css=css) as demo:
        # gr.Markdown(description)
        instance = get_gradio_instance(rag_chain)
        state = gr.State(instance)
        demo.load(
            None,
            None,
            js="""
      () => {
      const params = new URLSearchParams(window.location.search);
      if (!params.has('__theme')) {
        params.set('__theme', 'dark');
        window.location.search = params.toString();
      }
      }""",
        )
        gr.HTML(value=html_title)
        with gr.Row():
            with gr.Column(scale=4):
                video = gr.Video(height=512, width=512, elem_id="video", interactive=False )
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                            elem_id="chatbot", label="Multimodal RAG Chatbot", height=512,
                    )
                with gr.Row():
                    with gr.Column(scale=8):
                        # textbox.render()
                        textbox = gr.Dropdown(
                            dropdown_list,
                            allow_custom_value=True,
                            # show_label=False,
                            # container=False,
                            label="Query",
                            info="Enter your query here or choose a sample from the dropdown list!"
                        )
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=True
                        )
                with gr.Row(elem_id="buttons") as button_row:
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        btn_list = [clear_btn]
        
        clear_btn.click(
            clear_history, [state], [state, chatbot, textbox, video] + btn_list
        )
        submit_btn.click(
            add_text,
            [state, textbox],
            [state, chatbot, textbox,] + btn_list,
        ).then(
            http_bot,
            [state],
            [state, chatbot, video] + btn_list,
        )
    return demo

