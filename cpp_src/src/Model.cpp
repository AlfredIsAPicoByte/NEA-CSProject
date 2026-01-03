#include "Model.h"

Model::Model(const char* file)
{
    LoadModel(std::string(file));
}

void Model::LoadModel(const std::string& filePath)
{
    namespace fs = std::filesystem;
    try {
        fs::path requested(filePath);

        // Try a few likely candidates relative to the running CWD
        std::vector<fs::path> candidates = {
            fs::current_path() / requested,
            fs::current_path() / "models" / requested,
        };

        fs::path resolved_path;
        for (auto &c : candidates) {
            if (fs::exists(c)) { resolved_path = fs::canonical(c); break; }
        }

        // fallback: walk up parent directories
        if (resolved_path.empty()) {
            fs::path root = fs::current_path();
            int maxUp = 6;
            for (int depth = 0; depth < maxUp && !root.empty(); ++depth) {
                fs::path candidate = root / "cpp_src" / "src" / "models" / requested.filename();
                if (fs::exists(candidate)) { resolved_path = fs::canonical(candidate); break; }

                for (auto &entry : fs::recursive_directory_iterator(root)) {
                    if (!entry.is_regular_file()) continue;
                    if (entry.path().filename() == requested.filename()) { resolved_path = entry.path(); break; }
                }
                if (!resolved_path.empty()) break;

                if (root.has_parent_path()) root = root.parent_path(); else break;
            }
        }

        if (resolved_path.empty()) {
            throw std::runtime_error("Model file not found: " + requested.string());
        }

        AppendMessage("Model path: " + resolved_path.string());

        // Store resolved paths
        this->modelPath = resolved_path;
        this->file = filePath; // Store the original request string or resolved path, up to preference

        // Parse modelJSON
        std::string text = get_file_contents(resolved_path.string().c_str());
        try {
            modelJSON = json::parse(text);
        } catch (const json::parse_error& pex) {
            throw std::runtime_error(std::string("modelJSON parse error: ") + pex.what());
        }

        if (!modelJSON.contains("asset")) {
            throw std::runtime_error("Model modelJSON missing 'asset' section");
        }
        AppendMessage("Loaded model generator: " + modelJSON["asset"]["generator"].get<std::string>());

        // Load Binary Data
        data = getData();

        // Traverse all nodes
        traverseNode(0);

    } catch (const std::exception& e) {
        AppendError(std::string("Error loading model '") + filePath + "': " + e.what());
        throw; 
    }
}
void Model::loadMesh(unsigned int indMesh)
{
    // Get all accessor indices
    unsigned int posAccInd = modelJSON["meshes"][indMesh]["primitives"][0]["attributes"]["POSITION"];
    unsigned int normalAccInd = modelJSON["meshes"][indMesh]["primitives"][0]["attributes"]["NORMAL"];
    unsigned int texAccInd = modelJSON["meshes"][indMesh]["primitives"][0]["attributes"]["TEXCOORD_0"];
    unsigned int indAccInd = modelJSON["meshes"][indMesh]["primitives"][0]["indices"];

    // Use accessor indices to get all vertices components
    std::vector<float> posVec = getFloats(modelJSON["accessors"][posAccInd]);
    std::vector<glm::vec3> positions = groupFloatsVec3(posVec);
    std::vector<float> normalVec = getFloats(modelJSON["accessors"][normalAccInd]);
    std::vector<glm::vec3> normals = groupFloatsVec3(normalVec);
    std::vector<float> texVec = getFloats(modelJSON["accessors"][texAccInd]);
    std::vector<glm::vec2> texUVs = groupFloatsVec2(texVec);

    // Combine all the vertex components and also get the indices and textures
    std::vector<Vertex> vertices = assembleVertices(positions, normals, texUVs);
    std::vector<GLuint> indices = getIndices(modelJSON["accessors"][indAccInd]);
    std::vector<Texture> textures = getTextures();

    // Compute tangents for normal mapping
    computeTangents(vertices, indices);

    // Combine the vertices, indices, and textures into a mesh
    meshes.push_back(Mesh(vertices, indices, textures));

    // Check for missing UVs
    bool hasUV = false;
    for (auto &v : vertices) { if (v.TexCoords.x != 0.0f || v.TexCoords.y != 0.0f) { hasUV = true; break; } }
    if (!hasUV) {
        AppendMessage("Warning: mesh has no UVs; textures will not apply.");
    }
}

void Model::traverseNode(unsigned int nextNode, glm::mat4 matrix)
{
    // Current node
    json node = modelJSON["nodes"][nextNode];

    // Get translation if it exists
    glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f);
    if (node.contains("translation"))
    {
        float transValues[3];
        for (unsigned int i = 0; i < node["translation"].size(); i++)
            transValues[i] = (node["translation"][i]);
        translation = glm::make_vec3(transValues);
    }
    // Get quaternion if it exists
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.contains("rotation"))
    {
        float rotValues[4] =
        {
            node["rotation"][3],
            node["rotation"][0],
            node["rotation"][1],
            node["rotation"][2]
        };
        rotation = glm::make_quat(rotValues);
    }
    // Get scale if it exists
    glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);
    if (node.contains("scale"))
    {
        float scaleValues[3];
        for (unsigned int i = 0; i < node["scale"].size(); i++)
            scaleValues[i] = (node["scale"][i]);
        scale = glm::make_vec3(scaleValues);
    }
    // Get matrix if it exists
    glm::mat4 matNode = glm::mat4(1.0f);
    if (node.contains("matrix"))
    {
        float matValues[16];
        for (unsigned int i = 0; i < node["matrix"].size(); i++)
            matValues[i] = (node["matrix"][i]);
        matNode = glm::make_mat4(matValues);
    }

    // Initialize matrices
    glm::mat4 trans = glm::mat4(1.0f);
    glm::mat4 rot = glm::mat4(1.0f);
    glm::mat4 sca = glm::mat4(1.0f);

    // Use translation, rotation, and scale to change the initialized matrices
    trans = glm::translate(trans, translation);
    rot = glm::mat4_cast(rotation);
    sca = glm::scale(sca, scale);

    // Multiply all matrices together
    glm::mat4 matNextNode = matrix * matNode * trans * rot * sca;

    // Check if the node contains a mesh and if it does load it
    if (node.contains("mesh"))
    {
        matricesMeshes.push_back(matNextNode);
        loadMesh(node["mesh"]);
    }

    // Check if the node has children
    if (node.contains("children"))
    {
        for (unsigned int i = 0; i < node["children"].size(); i++)
            traverseNode(node["children"][i], matNextNode);
    }
}

std::vector<unsigned char> Model::getData()
{
    // Use filesystem to resolve buffer uri relative to the model file location
    std::string uri = modelJSON["buffers"][0]["uri"];
    std::filesystem::path binPath = modelPath.parent_path() / uri;
    AppendMessage("Resolved binary path: " + binPath.string());

    if (!std::filesystem::exists(binPath)) {
        throw std::runtime_error("Failed to open file: " + binPath.string());
    }

    // Read binary file
    std::ifstream infile(binPath, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open file: " + binPath.string());
    }

    infile.seekg(0, std::ios::end);
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer;
    buffer.resize(static_cast<size_t>(size));
    if (!infile.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read binary file: " + binPath.string());
    }

    return buffer;
}

std::vector<float> Model::getFloats(json accessor)
{
    std::vector<float> floatVec;

    // Get properties from the accessor
    unsigned int buffViewInd = accessor.value("bufferView", 0);
    unsigned int count = accessor["count"];
    unsigned int accByteOffset = accessor.value("byteOffset", 0);
    std::string type = accessor["type"];

    // Get properties from the bufferView
    json bufferView = modelJSON["bufferViews"][buffViewInd];
    unsigned int byteOffset = bufferView["byteOffset"];

    // Interpret the type
    unsigned int numPerVert;
    if (type == "SCALAR") numPerVert = 1;
    else if (type == "VEC2") numPerVert = 2;
    else if (type == "VEC3") numPerVert = 3;
    else if (type == "VEC4") numPerVert = 4;
    else throw std::invalid_argument("Type is invalid (not SCALAR, VEC2, VEC3, or VEC4)");

    // Go over all the bytes in the data
    unsigned int beginningOfData = byteOffset + accByteOffset;
    unsigned int lengthOfData = count * 4 * numPerVert;
    for (unsigned int i = beginningOfData; i < beginningOfData + lengthOfData; i += 4)
    {
        unsigned char bytes[] = { data[i], data[i + 1], data[i + 2], data[i + 3] };
        float value;
        std::memcpy(&value, bytes, sizeof(float));
        floatVec.push_back(value);
    }

    return floatVec;
}

std::vector<GLuint> Model::getIndices(json accessor)
{
    std::vector<GLuint> indices;

    // Get properties from the accessor
    unsigned int buffViewInd = accessor.value("bufferView", 0);
    unsigned int count = accessor["count"];
    unsigned int accByteOffset = accessor.value("byteOffset", 0);
    unsigned int componentType = accessor["componentType"];

    // Get properties from the bufferView
    json bufferView = modelJSON["bufferViews"][buffViewInd];
    unsigned int byteOffset = bufferView["byteOffset"];

    // Get indices with regards to their type: unsigned int, unsigned short, or short
    unsigned int beginningOfData = byteOffset + accByteOffset;
    if (componentType == 5125) // unsigned int
    {
        for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 4; i += 4)
        {
            unsigned char bytes[] = { data[i], data[i + 1], data[i + 2], data[i + 3] };
            unsigned int value;
            std::memcpy(&value, bytes, sizeof(unsigned int));
            indices.push_back((GLuint)value);
        }
    }
    else if (componentType == 5123) // unsigned short
    {
        for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 2; i += 2)
        {
            unsigned char bytes[] = { data[i], data[i + 1] };
            unsigned short value;
            std::memcpy(&value, bytes, sizeof(unsigned short));
            indices.push_back((GLuint)value);
        }
    }
    else if (componentType == 5122) // short
    {
        for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 2; i += 2)
        {
            unsigned char bytes[] = { data[i], data[i + 1] };
            short value;
            std::memcpy(&value, bytes, sizeof(short));
            indices.push_back((GLuint)value);
        }
    }

    return indices;
}

std::vector<Texture> Model::getTextures()
{
    std::vector<Texture> textures;

    // file is now std::string, so no need to cast
    std::string fileDirectory = file.substr(0, file.find_last_of('/') + 1);
    AppendMessage("Model directory: " + fileDirectory);

    // Go over all images
    for (unsigned int i = 0; i < modelJSON["images"].size(); i++)
    {
        std::string texPath = modelJSON["images"][i]["uri"];

        // Check if the texture has already been loaded
        bool skip = false;
        for (unsigned int j = 0; j < loadedTexName.size(); j++)
        {
            if (loadedTexName[j] == texPath)
            {
                textures.push_back(loadedTex[j]);
                skip = true;
                break;
            }
        }

        // If the texture has been loaded, skip this
        if (!skip)
        {
            // Load diffuse texture
            if (texPath.find("baseColor") != std::string::npos)
            {
                Texture diffuse = Texture((fileDirectory + texPath).c_str(), "diffuse", static_cast<GLuint>(loadedTex.size()));
                textures.push_back(diffuse);
                loadedTex.push_back(diffuse);
                loadedTexName.push_back(texPath);
             }
             // Load specular texture
             else if (texPath.find("metallicRoughness") != std::string::npos)
             {
                Texture specular = Texture((fileDirectory + texPath).c_str(), "specular", static_cast<GLuint>(loadedTex.size()));
                textures.push_back(specular);
                loadedTex.push_back(specular);
                loadedTexName.push_back(texPath);
             }
         }
     }

     return textures;
}

std::vector<Vertex> Model::assembleVertices
(
    std::vector<glm::vec3> positions,
    std::vector<glm::vec3> normals,
    std::vector<glm::vec2> texUVs
)
{
    std::vector<Vertex> vertices;
    for (int i = 0; i < positions.size(); i++)
    {
        vertices.push_back
        (
            Vertex
            {
                positions[i],
                normals[i],
                glm::vec3(1.0f, 1.0f, 1.0f),
                texUVs[i],
                // Tangents will be computed later
            }
        );
    }
    return vertices;
}

void Model::computeTangents(std::vector<Vertex>& verts, const std::vector<GLuint>& idx)
{
    std::vector<glm::vec3> accumT;
    accumT.resize(verts.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < idx.size(); i += 3)
    {
        Vertex &v0 = verts[idx[i + 0]];
        Vertex &v1 = verts[idx[i + 1]];
        Vertex &v2 = verts[idx[i + 2]];

        glm::vec3 edge1 = v1.Position - v0.Position;
        glm::vec3 edge2 = v2.Position - v0.Position;

        glm::vec2 deltaUV1 = v1.TexCoords - v0.TexCoords;
        glm::vec2 deltaUV2 = v2.TexCoords - v0.TexCoords;

        float denom = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
        float f = (std::fabs(denom) > 1e-8f) ? 1.0f / denom : 0.0f;

        glm::vec3 tangent = f * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);
        accumT[idx[i + 0]] += tangent;
        accumT[idx[i + 1]] += tangent;
        accumT[idx[i + 2]] += tangent;
    }

    for (size_t vi = 0; vi < verts.size(); ++vi)
    {
        glm::vec3 T = accumT[vi];
        glm::vec3 N = verts[vi].Normal;

        if (glm::length2(T) < 1e-8f) {
            T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 0.0f, 1.0f)));
            if (glm::length2(T) < 1e-6f)
                T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 1.0f, 0.0f)));
        } else {
            T = glm::normalize(T - N * glm::dot(N, T));
        }

        glm::vec3 B = glm::cross(N, T);
        float handedness = (glm::dot(B, glm::cross(N, T)) < 0.0f) ? -1.0f : 1.0f;
        verts[vi].Tangent = glm::vec4(T, handedness);
    }
}

std::vector<glm::vec2> Model::groupFloatsVec2(std::vector<float> floatVec)
{
    const size_t floatsPerVector = 2;
    std::vector<glm::vec2> vectors;
    vectors.reserve(floatVec.size() / floatsPerVector);

    for (size_t i = 0; i + (floatsPerVector - 1) < floatVec.size(); i += floatsPerVector)
    {
        vectors.emplace_back(floatVec[i], floatVec[i + 1]);
    }
    return vectors;
}

std::vector<glm::vec3> Model::groupFloatsVec3(std::vector<float> floatVec)
{
    const size_t floatsPerVector = 3;
    std::vector<glm::vec3> vectors;
    vectors.reserve(floatVec.size() / floatsPerVector);

    for (size_t i = 0; i + (floatsPerVector - 1) < floatVec.size(); i += floatsPerVector)
    {
        vectors.emplace_back(floatVec[i], floatVec[i + 1], floatVec[i + 2]);
    }
    return vectors;
}

std::vector<glm::vec4> Model::groupFloatsVec4(std::vector<float> floatVec)
{
    const size_t floatsPerVector = 4;
    std::vector<glm::vec4> vectors;
    vectors.reserve(floatVec.size() / floatsPerVector);

    for (size_t i = 0; i + (floatsPerVector - 1) < floatVec.size(); i += floatsPerVector)
    {
        vectors.emplace_back(floatVec[i], floatVec[i + 1], floatVec[i + 2], floatVec[i + 3]);
    }
    return vectors;
}

std::vector<Mesh>& Model::GetMeshes()
{
    return meshes;
}

glm::mat4 Model::GetModelMatrixForMesh(unsigned int meshIndex) const
{
    if (meshIndex >= matricesMeshes.size())
    {
        throw std::out_of_range("Mesh index out of range in GetModelMatrixForMesh");
    }
    return matricesMeshes[meshIndex];
}

std::vector<glm::mat4> Model::GetModelMatricesForAllMeshes() const
{
    return matricesMeshes;
}

void Model::SetModelMatrixForMesh(unsigned int meshIndex, const glm::mat4& modelMatrix)
{
    if (meshIndex >= matricesMeshes.size())
    {
        throw std::out_of_range("Mesh index out of range in SetModelMatrixForMesh");
    }
    matricesMeshes[meshIndex] = modelMatrix;
}

void Model::SetModelMatricesForAllMeshes(const std::vector<glm::mat4>& modelMatrices)
{
    if (modelMatrices.size() != matricesMeshes.size())
    {
        throw std::invalid_argument("Size of modelMatrices does not match number of meshes in SetModelMatricesForAllMeshes");
    }
    matricesMeshes = modelMatrices;
}

void Model::CleanUp() {
    
}

void Model::SerializeFields(json& j) const {
    // 1. Save the file path so we can reload the geometry later
    j["file_path"] = file;
	
    // 2. Save the current state of all mesh matrices
    // We need to flatten the matrices into arrays of floats
    std::vector<std::vector<float>> matricesData;
    for (const auto& mat : matricesMeshes) {
        const float* pSource = glm::value_ptr(mat);
        matricesData.emplace_back(pSource, pSource + 16);
    }
    j["matrices"] = matricesData;
    j["constructor"] = "ModelRaw";
}

void Model::DeserializeFields(const json& j) {
    // 1. Load the Geometry from file
    if (j.contains("file_path")) {
        std::string path = j["file_path"];
        LoadModel(path); // This re-populates 'meshes' and 'matricesMeshes'
    }
	
    // 2. Restore transformations
    // (If the user moved specific parts of the model, we overwrite the defaults here)
    if (j.contains("matrices")) {
        const auto& matricesData = j["matrices"];
        
        // Ensure we don't go out of bounds if the file changed
        size_t count = std::min(matricesData.size(), matricesMeshes.size());
        
        for (size_t i = 0; i < count; i++) {
            std::vector<float> matRaw = matricesData[i];
            if (matRaw.size() == 16) {
                matricesMeshes[i] = glm::make_mat4(matRaw.data());
            }
        }
    }
}