from datetime import datetime

import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import sys
#from config import token
from server import *

print(sys.argv)
token = sys.argv[1]

logging.basicConfig(level=logging.INFO)

storage = MemoryStorage()
bot = Bot(token=token)
dp = Dispatcher(bot, storage=storage)


class Form(StatesGroup):
    resolution = State()  # Super resolution mode
    first_style = State()  # Style transfer mode -- style photo
    second_style = State()  # Style transfer mode -- content photo


@dp.message_handler(commands=['start', 'help', 'menu'], state='*')
@dp.message_handler(text=['Меню'], state='*')
async def greetings(message: types.Message, state: FSMContext):
    print(message.from_user.id)
    await state.finish()
    username = ' '.join(name for name in [message.from_user.first_name, message.from_user.last_name] if name)
    me = await bot.get_me()
    bot_text = (
        f'Привет, {username}!\n\n'
        f'Меня зовут {me.first_name}. Я спешу оказать тебе услугу и нарисовать для тебя что-нибудь '
        f'прекрасное 🌄\n\nМожешь выбрать чем мы займемся в меню снизу'
    )
    await message.answer(bot_text, reply_markup=create_menu())


@dp.message_handler(text=['Super resolution', 'Style transfer'], state=None)
async def handler(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['function'] = message.text

    bot_text = ''
    if message.text == 'Super resolution':
        bot_text = (
            'Если ты пришлешь мне фотографию с маленьким разрешением 🏙, то я ее увеличу в 4 раза, а качество останется'
            ' на высоком уровне👍\n\n🔹 Пришли фотографию, лучше в виде документа без сжатия'
        )
        await Form.resolution.set()

    elif message.text == 'Style transfer':
        bot_text = (
            'Если ты пришлешь мне две свои любимые фотографии 📸, то я смогу сделать одну из них, похожей на другую '
            '🏙 👉 🌆\n\n🔻 Пришли мне фотографию, на которую будет похож результат'
        )
        await Form.first_style.set()
    await message.answer(bot_text, reply_markup=create_menu('Меню'))


@dp.message_handler(content_types=['photo', 'document'], state=Form.resolution)
async def handler(message: types.Message):
    print(message.from_user.id, 'Super resolution')
    file_id = message.document.file_id if message.document else message.photo[-1].file_id
    image = await download_image(bot, file_id)

    name = '{}_{}{}{}_{}{}{}.jpg'.format(message.chat.id, *datetime.now().timetuple())
    bot_text = 'Отлично! 👨‍🎨\n\nМне нужно немного времени 🚧⚙\nЯ напишу тебе, когда закончу'
    await message.answer(text=bot_text)

    image.save('./result/super_resolution/LR/{}'.format(name))
    result = await super_resolution(image)
    result.save('./result/super_resolution/SR/{}'.format(name))

    await sendback_result(message, result)
    await user_feedback(message)


@dp.message_handler(content_types=['photo', 'document'], state=Form.first_style)
async def handler(message: types.Message, state: FSMContext):
    print(message.from_user.id, 'Style transfer: style')
    file_id = message.document.file_id if message.document else message.photo[-1].file_id
    image = await download_image(bot, file_id)

    async with state.proxy() as data:
        data['photo_name'] = '{}_{}{}{}_{}{}{}.jpg'.format(message.chat.id, *datetime.now().timetuple())
        name = data['photo_name']

    image.save('./result/style_transfer/style/{}'.format(name), format='JPEG', quality='keep')

    await Form.second_style.set()
    bot_text = '🔻 Хорошо. Теперь пришли фотографию, которую будем менять, лучше документом без сжатия'
    await message.answer(text=bot_text)


@dp.message_handler(content_types=['photo', 'document'], state=Form.second_style)
async def handler(message: types.Message, state: FSMContext):
    print(message.from_user.id, 'Style transfer: content')
    file_id = message.document.file_id if message.document else message.photo[-1].file_id
    image = await download_image(bot, file_id)

    async with state.proxy() as data:
        name = data['photo_name']

    bot_text = 'Отлично! 👨‍🎨\n\nМне нужно немного времени 🚧⚙\nЯ напишу тебе, когда закончу'
    await message.answer(text=bot_text)

    image.save('./result/style_transfer/content/{}'.format(name), format='JPEG', quality='keep')
    result = await style_transfer(name)
    result.save('./result/style_transfer/result/{}'.format(name), format='JPEG', quality=95)

    await sendback_result(message, result)
    await user_feedback(message)
    await Form.first_style.set()


@dp.callback_query_handler(lambda call: call.data, state='*')
async def handler(call: types.callback_query.CallbackQuery):
    print(call.from_user.id, call.data)
    bot_text = '\nЕсли хочешь попробовать еще раз, пришли новое фото. 🗃\nЧтобы выйти в меню нажми /start'
    bot_text = call.data + bot_text
    await bot.edit_message_text(bot_text, call.from_user.id, call.message.message_id)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
