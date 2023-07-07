"""This module is responsible for parsing an HTTP message attributes JSON into its components
"""
import simplejson
import xml.etree.ElementTree as et
import case_id_assignment.utilities as util


def parse_file_data_from_xml(param):
    root = et.fromstring(param)
    filtered = [item for item in list(root.itertext()) if item.strip()]
    return filtered


def parse_method_and_file_data(row):
    session_class = row['session_class']
    message_attributes_is_str = False
    if session_class != 'http':
        return '', None, []
    if isinstance(row['MessageAttributes'], str):
        column_data = row['MessageAttributes'].replace("'", '"').replace('"1.0"', '1.0')
        message_attributes_is_str = True
    # print(column_data)
    xml = None
    starting_frame_number = ''
    try:
        message_attributes = simplejson.loads(column_data) if message_attributes_is_str else row['MessageAttributes']
        starting_frame_number = message_attributes.get('http.request_in', '')
    except simplejson.errors.JSONDecodeError:
        index = column_data.index('"http.file_data":') + len('"http.file_data":') + 2
        xml = column_data[index:-2].encode().decode('unicode_escape')

    try:
        xml = xml if xml else message_attributes['http.file_data']
    except KeyError:
        return 'KeyError'
    xml = xml if '"1.0"' in xml else xml.replace('1.0', '"1.0"')
    xml = xml.replace('\\xa', '')
    method = parse_file_data_from_xml(xml)
    method = method if method else []
    return util.first_item(method), starting_frame_number, method

def parse_message_attributes(row):
    session_class = row['session_class']
    if session_class != 'http':
        return '', None, []