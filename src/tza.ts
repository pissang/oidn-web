// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Helper types and functions
export type TensorLayout = 'x' | 'oihw';

export type DataType = 'Float32' | 'Float16';

export class TensorDesc {
  dims: number[] = [];
  paddedDims: number[] = [];
  layout: TensorLayout = 'x';
  dataType: DataType = 'Float32';

  getByteSize(): number {
    let size = 1;
    for (const dim of this.paddedDims) {
      size *= dim;
    }
    if (this.dataType === 'Float32') {
      size *= 4;
    } else if (this.dataType === 'Float16') {
      size *= 2;
    }
    return size;
  }
}

export class HostTensor {
  constructor(public desc: TensorDesc, public data: Uint8Array) {}
}

class DataViewReader {
  offset: number = 0;

  constructor(private _view: DataView) {}

  read(size: number): number {
    const view = this._view;
    const offset = this.offset;
    this.offset += size;
    switch (size) {
      case 1:
        return view.getUint8(offset);
      case 2:
        return view.getUint16(offset, true);
      case 4:
        return view.getUint32(offset, true);
      case 8:
        // TODO
        return Number(view.getBigUint64(offset, true));
      default:
        throw new Error('unsupported read size');
    }
  }
}

export function parseTZA(buffer: ArrayBuffer): Map<string, HostTensor> {
  const input = new Uint8Array(buffer);
  const viewReader = new DataViewReader(new DataView(buffer));

  // Parse the magic value
  const magic = viewReader.read(2);
  if (magic !== 0x41d7) {
    throw new Error('invalid or corrupted weights blob');
  }

  // Parse the version
  const majorVersion = viewReader.read(1);
  const minorVersion = viewReader.read(1);
  if (majorVersion !== 2) {
    throw new Error('unsupported weights blob version');
  }

  // Parse the table offset and jump to the table
  const tableOffset = viewReader.read(8);
  viewReader.offset = tableOffset;

  // Parse the number of tensors
  const numTensors = viewReader.read(4);

  // Parse the tensors
  const tensorMap = new Map<string, HostTensor>();
  for (let i = 0; i < numTensors; ++i) {
    const tensorDesc = new TensorDesc();

    // Parse the name
    const nameLen = viewReader.read(2);
    const name = new TextDecoder().decode(
      input.subarray(viewReader.offset, viewReader.offset + nameLen)
    );
    viewReader.offset += nameLen;

    // Parse the number of dimensions
    const ndims = viewReader.read(1);

    // Parse the shape of the tensor
    for (let j = 0; j < ndims; ++j) {
      tensorDesc.dims.push(viewReader.read(4));
    }

    tensorDesc.paddedDims = [...tensorDesc.dims];

    // Parse the layout of the tensor
    const layoutStr = new TextDecoder().decode(
      input.subarray(viewReader.offset, viewReader.offset + ndims)
    );
    if (layoutStr === 'oihw') {
      tensorDesc.layout = 'oihw';
    }

    viewReader.offset += ndims;

    // Parse the data type of the tensor
    const dataType = String.fromCharCode(viewReader.read(1));
    if (dataType === 'f') {
      tensorDesc.dataType = 'Float32';
    } else if (dataType === 'h') {
      tensorDesc.dataType = 'Float16';
    } else {
      throw new Error('invalid tensor data type');
    }

    // Parse the offset to the tensor data
    const tensorOffset = viewReader.read(8);
    const tensorData = input.subarray(
      tensorOffset,
      tensorOffset + tensorDesc.getByteSize()
    );

    // Add the tensor to the map
    tensorMap.set(name, new HostTensor(tensorDesc, tensorData));
  }

  return tensorMap;
}
