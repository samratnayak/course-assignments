Document layout samples (gibberish content)
==========================================

These files show how each optional DOCX/PDF layout looks. They are not real CVs.

Generate them (after installing dependencies, e.g. pip install -r requirements.txt):

  cd Codebase
  python build_style_samples.py

You will get:
  sample_default.docx
  sample_sidebar_modern.docx
  sample_classic_serif.docx

During conversational refinement, the app prints the full path to this folder so you
can open these samples and pick a layout by number or id.
