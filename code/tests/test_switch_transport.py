import re
from src.switch_transport import Connection


class FakeSocket:
    def __init__(self, chunks):
        self.chunks, self.sent = list(chunks), []

    def recv(self, size):
        return self.chunks.pop(0) if self.chunks else b''

    def sendall(self, value):
        self.sent.append(value)


def connection(chunks):
    conn = Connection.__new__(Connection)
    conn.sock = FakeSocket(chunks)
    conn.pending = b''
    conn.enabled_remote = set()
    return conn


def test_pagination_every_possible_tcp_split():
    page = b'first page\r\n---- More ----'
    tail = b'\r\nlast page\r\n<H3C>'
    for boundary in range(1, len(page)):
        conn = connection([page[:boundary], page[boundary:], tail])
        result = conn.read(r'<H3C>\s*$')
        assert 'last page' in result
        assert conn.sock.sent == [b' '], boundary


def test_bytewise_two_pages_are_each_acknowledged_once():
    reply = b'first\r\n---- More ----\r\nsecond\r\n---- More ----\r\n<H3C>'
    conn = connection([bytes([x]) for x in reply])
    assert re.search(r'<H3C>$', conn.read(r'<H3C>$'))
    assert conn.sock.sent == [b' ', b' ']


def test_iac_split_negotiation_and_no_reack_loop():
    reply = b'\xff\xfb\x01\xff\xfb\x01\xff\xfd\x1f\xff\xfa\x18abc\xff\xf0\r\n<H3C>'
    conn = connection([bytes([x]) for x in reply])
    assert conn.read(r'<H3C>$') == '\r\n<H3C>'
    assert conn.sock.sent == [b'\xff\xfd\x01', b'\xff\xfc\x1f']
    assert conn.pending == b''
