

def GenerateMarkdownTable(head_list, index_list, table_corner, array_list):
    # array_list sort row first
    if len(head_list) * len(index_list) != len(array_list):
        print('the table size is not equal to the head_list and the index list')
        return

    # Add head
    markdown = '| '
    markdown += (table_corner + ' |')
    for head in head_list:
        markdown += (' ' + head + ' |')
    markdown += '\n'

    # Add head line
    markdown += '| '
    for index in range(len(head_list) + 1):
        markdown += ' ---- |'
    markdown += '\n'

    # Add rows
    for row_index in range(len(index_list)):
        markdown += '| '
        markdown += (' ' + index_list[row_index] + ' |')
        for col_index in range(len(head_list)):
            markdown += (' ' + array_list[row_index * len(head_list) + col_index] + ' |')
        markdown += '\n'

    return markdown