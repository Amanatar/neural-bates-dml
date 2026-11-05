import base64
import os

with open('d:/neural-bates-dml/artifacts/bates_dml_weights.bin', 'rb') as f:
    raw = f.read()

b64 = base64.b64encode(raw).decode('ascii')

with open('d:/neural-bates-dml/weights_data.js', 'w', encoding='utf-8') as f:
    f.write('// Auto-generated packed float32 weights for Neural Bates DML\n')
    f.write('var BATES_DML_WEIGHTS_B64 = "' + b64 + '";\n')
    f.write('if (typeof module !== "undefined") { module.exports = { BATES_DML_WEIGHTS_B64 }; }\n')
    f.write('if (typeof window !== "undefined") { window.BATES_DML_WEIGHTS_B64 = BATES_DML_WEIGHTS_B64; }\n')

print('Wrote weights_data.js:', os.path.getsize('d:/neural-bates-dml/weights_data.js') / 1024, 'KB')
