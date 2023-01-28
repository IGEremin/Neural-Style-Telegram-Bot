import io
from PIL import Image
import numpy as np
from aiogram import types
import onnxruntime as ort

# --- SERVER ---

__all__ = ['create_menu', 'download_image', 'user_feedback', 'sendback_result', 'super_resolution', 'style_transfer']


def create_menu(*args):
    if not args:
        args = ('Super resolution', 'Style transfer')
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(*args)
    return markup


async def download_image(bot, file_id: str) -> Image.Image:
    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    image = await bot.download_file(file_path)
    image = Image.open(image, formats=['jpeg'])
    return image


async def user_feedback(message: types.Message):
    bot_text = 'Как тебе результат? ✨'
    markup = types.InlineKeyboardMarkup().row(
        types.InlineKeyboardButton('❤️', callback_data='❤️ lovely!'),
        types.InlineKeyboardButton('😄', callback_data='😄 funny!'),
        types.InlineKeyboardButton('🗿', callback_data='🗿 dummy!')
    )
    await message.answer(text=bot_text, reply_markup=markup)


async def sendback_result(message: types.Message, image: Image.Image):
    result = io.BytesIO()
    result.name = 'image.jpeg'
    image.save(result, format='JPEG')
    await message.answer_photo(result.getbuffer())
    result.seek(0)
    await message.answer_document(result)


async def super_resolution(image: Image) -> Image.Image:
    image = np.array(image)
    image = image / 127.5 - 1
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = image[np.newaxis, :, :, :]

    # Load SRGAN model and weights
    # Forward input image through the model
    ort_session = ort.InferenceSession("./models/SRGAN/srgan.onnx")
    output = ort_session.run(None, {'lr': image})

    # Convert output image to PIL.Image
    output = output[0].squeeze(0)
    output = (output + 1.0) * 127.5
    output = output.astype(np.uint8).transpose(1, 2, 0)
    output = Image.fromarray(output)
    return output


async def style_transfer(name: str) -> Image.Image:
    # Load style image, convert to torch.Tensor, convert RGB to BGR
    style_image = load_image('./result/style_transfer/style/{}'.format(name), 512)

    # Load content image
    content_image = load_image('./result/style_transfer/content/{}'.format(name))

    # Forward content image to STGAN
    ort_session = ort.InferenceSession("./models/STGAN/stgan.onnx")
    outputs = ort_session.run(None, {'xs': style_image, 'xc': content_image})

    output = outputs[0].squeeze(0)
    output = chanel_revert(output)
    output = output.clip(0, 255).transpose(1, 2, 0).astype(np.uint8)
    output = Image.fromarray(output)
    return output


def load_image(path: str, size: int = None):
    image = Image.open(path)
    if size:
        image = image.resize((size, size))
    image = np.array(image).transpose((2, 0, 1))
    image = chanel_revert(image).astype(np.float32)
    image = image[np.newaxis, :, :, :]
    return image


def chanel_revert(image):
    (a, b, c) = np.split(image, 3)
    image = np.concatenate([c, b, a], axis=0)
    return image
