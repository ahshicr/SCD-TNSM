import fitz
import pytest
from verify_reproduced_figures import compare_pdf


def pdf(path,text='Mean = 1.23',position=(40,60),metadata=None):
    with fitz.open() as document:
        page=document.new_page(width=240,height=150)
        page.insert_text(position,text)
        document.set_metadata(metadata or {})
        document.save(path)


def test_figure_comparison_ignores_creation_metadata_but_preserves_visible_evidence(tmp_path):
    a,b=tmp_path/'a.pdf',tmp_path/'b.pdf'
    pdf(a,metadata={'creationDate':'D:20260916000000'})
    pdf(b,metadata={'creationDate':'D:20261105000000'})
    report=compare_pdf(a,b)
    assert report['reference_sha256']!=report['recomputed_sha256']
    assert report['pages'][0]['identical']


def test_figure_comparison_rejects_changed_value(tmp_path):
    a,b=tmp_path/'a.pdf',tmp_path/'b.pdf'
    pdf(a)
    pdf(b,text='Mean = 1.24')
    with pytest.raises(AssertionError,match='visible text'):
        compare_pdf(a,b)


def test_figure_comparison_rejects_changed_position_even_when_text_is_identical(tmp_path):
    a,b=tmp_path/'a.pdf',tmp_path/'b.pdf'
    pdf(a)
    pdf(b,position=(42,60))
    with pytest.raises(AssertionError,match='rendering changed'):
        compare_pdf(a,b)
