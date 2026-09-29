"""Small derived external bundle for the already-authorized Modal workspace."""
from pathlib import Path
import tarfile
import pandas as pd
from src.experiments.common import save_csv,save_json,sha256


def main():
    root = Path('outputs/generalization_upgrade')
    destination = root/'cloud_inputs'
    destination.mkdir(exist_ok=True)
    files = []
    for condition,source in [('original','original_matched.csv'),('normalized','normalized.csv')]:
        frame = pd.read_csv(root/'02_external_bias_audit'/source)
        for index,row in frame.iterrows():
            relative = row.sample_id if condition=='original' else row.sample_id.replace('deepfakeface_sd15/','normalized_external/',1)
            files.append((Path(row.image_path),relative))
            frame.loc[index,'image_path'] = relative
        save_csv(destination/f'{condition}.csv',frame)
    archive = destination/'external.tar'
    with tarfile.open(archive,'w') as handle:
        for path,relative in files:
            handle.add(path,arcname=relative,recursive=False)
    save_json(destination/'bundle.json',{'archive_sha256':sha256(archive),'files':len(files)})


if __name__=='__main__':
    main()
