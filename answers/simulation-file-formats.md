# Simulation File Formats

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Simulation data comes in four layers, each with its own formats. **Geometry:** STL (triangles only, no units, no topology, often not watertight), OBJ and PLY (meshes, optionally with attributes), and exact CAD in **STEP** or IGES. **Meshes and results:** **VTK** (legacy .vtk and the XML family .vtu/.vtp/.vtm that ParaView reads natively), **CGNS** (the CFD standard, built on HDF5), **Exodus II** (finite elements, built on netCDF), EnSight, and solver-native formats such as **OpenFOAM** case directories. **Solver input decks:** Nastran .bdf, Abaqus .inp, LS-DYNA .k, Ansys .cdb, Fluent .cas/.dat, which describe mesh, materials, loads and settings as text or binary records. **Training data at scale:** array formats with chunking and compression, **HDF5**, **Zarr** (cloud-friendly, one object per chunk), Parquet for tabular metadata, and sharded formats for streaming. Handling terabytes means converting once from the solver formats into a consistent, chunked, compressed, versioned store with per-sample metadata, keeping the originals, and streaming from object storage instead of loading everything into memory.

**On this page**

- [Geometry formats](#geometry-formats)
- [Mesh and result formats](#mesh-and-result-formats)
- [Solver input decks](#solver-input-decks)
- [Array formats for training data](#array-formats-for-training-data)
- [Handling terabytes of simulation files](#handling-terabytes-of-simulation-files)
- [Code: read a mesh, attach a field, save for ParaView](#code-read-a-mesh-attach-a-field-save-for-paraview)
- [In an interview](#in-an-interview)

## Geometry formats

| Format | Content | Watch out for |
| --- | --- | --- |
| **STL** (ASCII or binary) | A list of triangles, each with three vertex coordinates and a normal | No units, no colours or parts, **no shared vertices** (a "triangle soup", so every vertex is repeated about six times and must be welded), often not watertight, normals may be inconsistent. Binary STL is 84 bytes of header plus 50 bytes per triangle |
| **OBJ** | Vertices, faces (any polygon), normals, texture coordinates, groups | Indexing starts at 1; optional material files |
| **PLY** | Vertices and faces with arbitrary per-element properties | Flexible but variants differ |
| **STEP** (ISO 10303, AP203/AP214/AP242) | Exact B-rep CAD: NURBS surfaces, topology, assemblies, units, sometimes PMI (tolerances) | Needs a CAD kernel (Open CASCADE via CadQuery, pythonOCC or gmsh) to read; tessellate for learning |
| **IGES** | Older neutral CAD exchange | Surfaces without reliable topology; repair often needed |
| **3MF, glTF** | Modern mesh formats with units, materials and metadata | Less common in CAE |
| **Parasolid (.x_t), ACIS (.sat), native CAD** | Kernel-specific exact geometry | Proprietary readers |

**Practical rule:** keep the original CAD (STEP) as the source of truth, and generate meshes or point clouds from it with recorded tessellation settings, so a resampled geometry is reproducible. See [CAD models and CAD datasets](geometry-representations.md#cad-models-and-cad-datasets).

## Mesh and result formats

| Format | Content | Notes |
| --- | --- | --- |
| **VTK legacy (.vtk)** | Points, cells of any type, point and cell data | Simple; ASCII or binary |
| **VTK XML (.vtu unstructured, .vtp polydata, .vts/.vtr structured, .vtm multiblock, .pvtu parallel)** | The same, with compression and parallel pieces | ParaView's native format; best for exchanging and visualizing results |
| **CGNS** | Structured and unstructured zones, boundary conditions, flow solutions, time series; on HDF5 | The CFD community standard (NASA-led); supported by most CFD codes |
| **Exodus II** | Finite-element meshes with element blocks, node and side sets, time steps; on netCDF | Common in Sandia and MOOSE-based codes and many FE tools |
| **EnSight Gold** | Geometry and variables, transient, parallel | Common export from commercial CFD |
| **OpenFOAM case** | Directories per time step with ASCII or binary field files and a polyMesh folder | Read with ParaView's reader, `fluidfoam`, or convert with `foamToVTK` |
| **Fluent .cas/.dat, .cas.h5** | Mesh and solution (recent versions on HDF5) | Proprietary layout; export to EnSight, CGNS or VTK |
| **Nastran .op2, Abaqus .odb, LS-DYNA d3plot** | Structural results | Vendor or open readers (pyNastran, lasso-python for d3plot); .odb needs Abaqus's Python |
| **Gmsh .msh, MED, XDMF** | Meshes (and XDMF + HDF5 for results, as FEniCS uses) | Good open-source exchange formats |

**meshio** reads and writes most of these mesh formats in Python; **PyVista** wraps VTK for analysis and plotting; ParaView (with its Python interface, pvpython) handles almost everything for visualization.

## Solver input decks

An input deck is the whole simulation setup: mesh, element types, materials, sections, boundary conditions, loads, contact definitions, solver controls and requested outputs.

| Solver | Deck | Structure |
| --- | --- | --- |
| Nastran (MSC, NX, Optistruct) | .bdf / .dat / .nas | Fixed or free-field "cards" (GRID, CTETRA, MAT1, PSOLID, SPC, FORCE) |
| Abaqus | .inp | Keyword lines (*NODE, *ELEMENT, *MATERIAL, *STEP, *BOUNDARY) |
| LS-DYNA | .k / .key | Keyword cards (*NODE, *PART, *MAT_..., *CONTACT_...) |
| Ansys Mechanical APDL | .cdb, .dat, .inp | APDL commands and block-formatted data |
| OpenFOAM | system/, constant/, 0/ directories | Dictionaries (controlDict, fvSchemes, fvSolution, boundary conditions per field) |
| CalculiX | .inp | Abaqus-like keywords |

**Why they matter for ML.** They are the inputs that agents and data pipelines generate (templated decks with parameters substituted), and they contain the metadata a surrogate needs (load values, material properties, boundary conditions). Parse them with pyNastran, meshio (meshes only), or vendor APIs (PyAnsys, Abaqus scripting), and store the parameters explicitly next to each result. See [LLM agents for simulation](llm-agents-for-simulation.md).

## Array formats for training data

| Format | Strengths | Weaknesses |
| --- | --- | --- |
| **HDF5** | Hierarchical groups, chunked and compressed datasets, attributes, partial reads; mature, everywhere | One big file; parallel writes need MPI builds; awkward on object storage |
| **Zarr** | Same data model as HDF5, but each chunk is a separate object: ideal for cloud storage and parallel writes | Many small objects if chunks are small; newer (version 3 format) |
| **NetCDF4** | HDF5 underneath with conventions (CF) for dimensions and coordinates | Geared to gridded data |
| **Parquet / Arrow** | Columnar tables; great for per-sample metadata and scalar outputs | Not for large meshes |
| **NumPy .npy / .npz** | Trivial | No chunking, weak metadata; fine for small datasets |
| **WebDataset (tar shards), TFRecord, MDS** | Sequential streaming of shuffled shards from object storage | Random access is harder |

**Choices that matter more than the container:** float32 (or float16 for storage) instead of float64; chunk shapes aligned with how training reads data (one sample, or one spatial block of a sample); a fast compressor (Blosc/LZ4/Zstandard) for training data; consistent units and field names; and per-sample metadata (parameters, solver version, convergence, mesh statistics) stored alongside.

## Handling terabytes of simulation files

DrivAerNet++ is over 39 TB and The Well 15 TB; a company's archive of crash or CFD results is often larger. A practical pipeline:

1. **Inventory and validate.** Catalogue every case with its parameters, solver and version, convergence status and file checksums; flag failed and unconverged runs. Keep the catalogue as a table (Parquet or a database).
2. **Convert once, in parallel.** Read solver outputs with format-specific readers (in parallel jobs, one per case: Dask, Ray or a cluster scheduler) and write a canonical training format: chunked, compressed arrays per sample (Zarr or HDF5), consistent units and coordinate frames, only the fields and regions needed.
3. **Preprocess what training needs.** Surface extraction, point sampling, normals, signed distances, graph connectivity, normalization statistics, train/validation/test splits by design, all computed once and versioned.
4. **Keep the originals** in cheap storage; the converted dataset is a derived, reproducible artefact with the conversion code version recorded.
5. **Stream during training.** Read from object storage with prefetching and multiple workers, shuffle at the shard level, cache hot data on local NVMe; sample point subsets per step instead of loading full volume meshes. See [Feeding terabytes to the GPUs](scaling-to-large-meshes.md#feeding-terabytes-to-the-gpus).
6. **Version data with models.** A dataset version identifier in every training run, so a model can be traced to the exact samples it saw. See [Versioning data together with models](production-and-deployment.md#versioning-data-together-with-models).

## Code: read a mesh, attach a field, save for ParaView

Write a binary STL (geometry only), read it back with meshio, attach a point field (a pressure coefficient) and a cell field (triangle areas), save a VTU file that ParaView opens directly, and finally store 20 "simulations" in one chunked, compressed HDF5 file with per-sample metadata. Requires `pip install meshio h5py`.

```python
import os, tempfile
import numpy as np
import meshio, h5py

workdir = tempfile.mkdtemp()
# A sphere-like surface of triangles (latitude-longitude grid), standing in for a CAD export.
n_lat, n_lon = 40, 80
th, ph = np.meshgrid(np.linspace(0.05, np.pi - 0.05, n_lat), np.linspace(0, 2 * np.pi, n_lon, endpoint=False), indexing="ij")
V = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
idx = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
a, b = idx[:-1, :], np.roll(idx[:-1, :], -1, axis=1)
c, d = idx[1:, :], np.roll(idx[1:, :], -1, axis=1)
F = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])

# 1. A binary STL holds triangles only: no shared vertices, no units, no fields.
stl_path = os.path.join(workdir, "part.stl")
meshio.write(stl_path, meshio.Mesh(V, [("triangle", F)]), binary=True)
stl = meshio.read(stl_path)                                  # meshio merges the repeated vertices on reading
print(f"STL: {os.path.getsize(stl_path):,} bytes, {len(stl.points)} points, {len(stl.cells_dict['triangle'])} triangles after reading")

# 2. Attach fields and save as VTU for ParaView.
pts, tri = stl.points, stl.cells_dict["triangle"]
cp = 1 - 2.25 * (1 - pts[:, 0] ** 2)                         # potential-flow Cp on a sphere, flow along x
area = 0.5 * np.linalg.norm(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1)
mesh = meshio.Mesh(pts, [("triangle", tri)], point_data={"Cp": cp}, cell_data={"area": [area]})
vtu_path = os.path.join(workdir, "part_with_cp.vtu")
meshio.write(vtu_path, mesh)                                 # open this file in ParaView
back = meshio.read(vtu_path)
print(f"VTU: {os.path.getsize(vtu_path):,} bytes, point fields {list(back.point_data)}, cell fields {list(back.cell_data)}, "
      f"Cp round trip exact: {np.array_equal(back.point_data['Cp'], cp)}")

# 3. Many samples for training: one HDF5 file, chunked and compressed, with metadata per sample.
h5_path = os.path.join(workdir, "dataset.h5")
with h5py.File(h5_path, "w") as h5:
    for s in range(20):
        g = h5.create_group(f"sample_{s:04d}")
        g.create_dataset("points", data=pts.astype(np.float32), compression="gzip", chunks=True)
        g.create_dataset("cells", data=tri.astype(np.int32), compression="gzip", chunks=True)
        g.create_dataset("Cp", data=(cp * (1 + 0.01 * s)).astype(np.float32), compression="gzip", chunks=True)
        g.attrs.update({"solver": "example-solver 1.0", "inlet_velocity_m_s": 30.0 + s, "converged": True})
with h5py.File(h5_path, "r") as h5:
    sample = h5["sample_0007"]                               # partial read: only this sample is loaded
    print(f"HDF5: {os.path.getsize(h5_path):,} bytes for 20 samples; sample_0007 inlet velocity "
          f"{sample.attrs['inlet_velocity_m_s']} m/s, Cp dtype {sample['Cp'].dtype}, first Cp {sample['Cp'][0]:.4f}")
raw = 20 * (pts.astype(np.float32).nbytes + tri.astype(np.int32).nbytes + cp.astype(np.float32).nbytes)
print(f"uncompressed size of the same arrays: {raw:,} bytes")
```

Output:

```
STL: 312,084 bytes, 3200 points, 6240 triangles after reading
VTU: 69,202 bytes, point fields ['Cp'], cell fields ['area'], Cp round trip exact: True
HDF5: 1,047,561 bytes for 20 samples; sample_0007 inlet velocity 37.0 m/s, Cp dtype float32, first Cp -1.3315
uncompressed size of the same arrays: 2,521,600 bytes
```

The binary STL is exactly 84 + 50 × 6,240 bytes, and meshio welds its repeated vertices back to 3,200 shared points. The VTU file (compressed by default) is less than a quarter of the STL's size although it also stores two fields. The HDF5 store is less than half the raw array size and can be read one sample at a time, with the simulation parameters stored as attributes next to the data.

## In an interview

**Thirty-second answer to "Which file formats will you meet, and how do you handle terabytes of them?"** "Geometry comes as STL, which is a triangle soup with no units, or exact CAD in STEP. Meshes and results come as VTK or VTU for ParaView, CGNS for CFD, Exodus for finite elements, or solver-native formats like OpenFOAM cases, and setups as input decks like Nastran bdf, Abaqus inp or LS-DYNA keyword files. For terabytes I catalogue and validate every case, convert once in parallel into a chunked, compressed array store such as Zarr or HDF5 with consistent units and per-sample metadata, precompute what training needs, keep the originals, version the derived dataset, and stream shards from object storage during training."

**Likely follow-ups**

- *What is wrong with STL?* No units, no shared vertices, no topology or parts, often not watertight, inconsistent normals.
- *HDF5 or Zarr?* Zarr for cloud object storage and parallel writes (a chunk per object); HDF5 for local or HPC file systems and mature tooling.
- *How do you make conversions reproducible?* Version the conversion code and settings, record them with the derived data, and keep the originals.

## References

- [meshio](https://github.com/nschloe/meshio) and [PyVista](https://github.com/pyvista/pyvista)
- [VTK file formats](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) and [ParaView](https://www.paraview.org/)
- [CGNS standard](https://cgns.github.io/)
- [Exodus II](https://github.com/sandialabs/seacas)
- [HDF5 (The HDF Group)](https://www.hdfgroup.org/solutions/hdf5/), [h5py](https://docs.h5py.org/) and [Zarr](https://zarr.dev/)
- [pyNastran](https://github.com/SteveDoyle2/pyNastran)
- [OpenFOAM user guide: file format](https://doc.cfd.direct/openfoam/user-guide-v12/basic-file-format)
