import serial
import time

def send_hex_command(ser, hex_command):
    """
    在已打开的串口上发送十六进制命令并接收响应

    参数:
        ser (Serial): 已打开的串口对象
        hex_command (str): 十六进制命令字符串
    """
    try:
        # 将十六进制字符串转换为字节数据
        byte_command = bytes.fromhex(hex_command.replace(" ", ""))
        ser.write(byte_command)
        print(f"已发送命令（十六进制）: {hex_command}")

        # 等待并读取响应
        time.sleep(0.1)  # 根据设备响应时间调整
        response = ser.read_all()

        if response:
            hex_response = " ".join([f"{byte:02X}" for byte in response])
            print(f"收到响应（十六进制）: {hex_response}")
        else:
            print("未收到响应")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    port_name = 'COM3'  # 修改为你的串口号

    # 一次性打开串口
    ser = serial.Serial(
        port=port_name,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )
    print(f"成功连接到串口 {port_name}")

    try:
        # 连续发送多条命令
        send_hex_command(ser, "55 bb 06 02  00 02 00 0a")  # 设备工作状态到空闲

        send_hex_command(ser, "55 bb 06 02  00 02 01 0b")  # 循环刺激

        send_hex_command(ser, "55 BB 0D 02 01 04 00 14 14 00 00 04 00 02 42")  # 设置通道参数

        send_hex_command(ser, "55 bb 06 02  01 08 02 13")  # 切换到电流调节

        send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, "55 BB 07 02 01 06 00 03 13")  # 调节电流——1ma

        # send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        # send_hex_command(ser, "55 BB 07 02 01 06 00 05 15")  # 调节电流

        send_hex_command(ser, " 55 bb 06 02  01 08 03 14")  # 正常工作
        time.sleep(5)

        send_hex_command(ser, "55 BB 06 02 01 08 01 12")  # 暂停
        time.sleep(3)

        send_hex_command(ser, "55 BB 06 02 01 08 03 14")  # 恢复
        time.sleep(5)

        send_hex_command(ser, "55 BB 06 02 01 08 01 12")  # 暂停
        time.sleep(3)

        # send_hex_command(ser, "55 BB 06 02 01 08 01 12")  # 暂停
        # time.sleep(10)
        #
        # send_hex_command(ser, "55 BB 06 02 01 08 03 14")  # 恢复
        # time.sleep(10)
        #


    finally:
        ser.close()
        print("串口已关闭")

