#include "Model.h"

Model::Model(const char* file)
{
    namespace fs = std::filesystem;
	try {
		fs::path requested(file);

		// Try a few likely candidates relative to the running CWD
		std::vector<fs::path> candidates = {
			fs::current_path() / requested,
			fs::current_path() / "models" / requested,
			fs::current_path() / "cpp_src" / "src" / "models" / requested,
			fs::current_path().parent_path() / "cpp_src" / "src" / "models" / requested
		};

		fs::path model_path;
		for (auto &c : candidates) {
			if (fs::exists(c)) { model_path = fs::canonical(c); break; }
		}

		// fallback: walk up parent directories and search under each root for the filename
		if (model_path.empty()) {
			fs::path root = fs::current_path();
			int maxUp = 6;
			for (int depth = 0; depth < maxUp && !root.empty(); ++depth) {
				// try common models folder under this root
				fs::path candidate = root / "cpp_src" / "src" / "models" / requested.filename();
				if (fs::exists(candidate)) { model_path = fs::canonical(candidate); break; }

				// recursive search under this root (limited depth)
				for (auto &entry : fs::recursive_directory_iterator(root)) {
					if (!entry.is_regular_file()) continue;
					if (entry.path().filename() == requested.filename()) { model_path = entry.path(); break; }
				}
				if (!model_path.empty()) break;

				if (root.has_parent_path()) root = root.parent_path(); else break;
			}
		}

		if (model_path.empty()) {
			throw std::runtime_error(std::string("Model file not found: ") + requested.string());
		}

		AppendMessage("Model path: " + model_path.string());

		// store resolved model path for later (used to resolve .bin and images)
		this->modelPath = model_path;

		std::string text = get_file_contents(model_path.string().c_str());

        try {
            JSON = json::parse(text);
        } catch (const json::parse_error& pex) {
            throw std::runtime_error(std::string("JSON parse error: ") + pex.what());
        }

        if (!JSON.contains("asset")) {
            throw std::runtime_error("Model JSON missing 'asset' section");
        }
        AppendMessage("Loaded model generator: " + JSON["asset"]["generator"].get<std::string>());

        this->file = file;
        data = getData();

        // Traverse all nodes
        traverseNode(0);
    } catch (const std::exception& e) {
        AppendError(std::string("Error loading model '") + file + "': " + e.what());
        throw; // rethrow to be handled by caller
    }
}
 
Model::Model(Model&& other) noexcept
	: name(std::move(other.name)),
	  id(other.id),
	  file(other.file),
	  data(std::move(other.data)),
	  JSON(std::move(other.JSON)),
	  modelPath(std::move(other.modelPath)),
	  meshes(std::move(other.meshes)),
	  matricesMeshes(std::move(other.matricesMeshes)),
	  loadedTexName(std::move(other.loadedTexName)),
	  loadedTex(std::move(other.loadedTex))
{
	other.id = 0;
}

void Model::Draw(Shader& shader, Camera& camera)
{
    // Go over all meshes and draw each one
    for (unsigned int i = 0; i < meshes.size(); i++)
    {
        meshes[i].SetModelMatrix(matricesMeshes[i]);
        // call instance Draw to match Mesh API
        meshes[i].Draw(shader, camera);
    }
}

void Model::loadMesh(unsigned int indMesh)
{
	// Get all accessor indices
	unsigned int posAccInd = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["POSITION"];
	unsigned int normalAccInd = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["NORMAL"];
	unsigned int texAccInd = JSON["meshes"][indMesh]["primitives"][0]["attributes"]["TEXCOORD_0"];
	unsigned int indAccInd = JSON["meshes"][indMesh]["primitives"][0]["indices"];

	// Use accessor indices to get all vertices components
	std::vector<float> posVec = getFloats(JSON["accessors"][posAccInd]);
	std::vector<glm::vec3> positions = groupFloatsVec3(posVec);
	std::vector<float> normalVec = getFloats(JSON["accessors"][normalAccInd]);
	std::vector<glm::vec3> normals = groupFloatsVec3(normalVec);
	std::vector<float> texVec = getFloats(JSON["accessors"][texAccInd]);
	std::vector<glm::vec2> texUVs = groupFloatsVec2(texVec);

	// Combine all the vertex components and also get the indices and textures
	std::vector<Vertex> vertices = assembleVertices(positions, normals, texUVs);
	std::vector<GLuint> indices = getIndices(JSON["accessors"][indAccInd]);
	std::vector<Texture> textures = getTextures();

	// Compute tangents for normal mapping
	computeTangents(vertices, indices);

	// Combine the vertices, indices, and textures into a mesh
	meshes.push_back(Mesh(vertices, indices, textures));

	// after loading a mesh's vertices:
	bool hasUV = false;
	for (auto &v : vertices) { if (v.TexCoords.x != 0.0f || v.TexCoords.y != 0.0f) { hasUV = true; break; } }
	if (!hasUV) {
	    AppendMessage("Warning: mesh has no UVs; textures will not apply.");
	}
}

void Model::traverseNode(unsigned int nextNode, glm::mat4 matrix)
{
	// Current node
	json node = JSON["nodes"][nextNode];

	// Get translation if it exists
	glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f);
	if (node.find("translation") != node.end())
	{
		float transValues[3];
		for (unsigned int i = 0; i < node["translation"].size(); i++)
			transValues[i] = (node["translation"][i]);
		translation = glm::make_vec3(transValues);
	}
	// Get quaternion if it exists
	glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	if (node.find("rotation") != node.end())
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
	if (node.find("scale") != node.end())
	{
		float scaleValues[3];
		for (unsigned int i = 0; i < node["scale"].size(); i++)
			scaleValues[i] = (node["scale"][i]);
		scale = glm::make_vec3(scaleValues);
	}
	// Get matrix if it exists
	glm::mat4 matNode = glm::mat4(1.0f);
	if (node.find("matrix") != node.end())
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
	if (node.find("mesh") != node.end())
	{
		matricesMeshes.push_back(matNextNode);

		loadMesh(node["mesh"]);
	}

	// Check if the node has children, and if it does, apply this function to them with the matNextNode
	if (node.find("children") != node.end())
	{
		for (unsigned int i = 0; i < node["children"].size(); i++)
			traverseNode(node["children"][i], matNextNode);
	}
}

std::vector<unsigned char> Model::getData()
{
    // Use filesystem to resolve buffer uri relative to the model file location
    std::string uri = JSON["buffers"][0]["uri"];
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
	json bufferView = JSON["bufferViews"][buffViewInd];
	unsigned int byteOffset = bufferView["byteOffset"];

	// Interpret the type and store it into numPerVert
	unsigned int numPerVert;
	if (type == "SCALAR") numPerVert = 1;
	else if (type == "VEC2") numPerVert = 2;
	else if (type == "VEC3") numPerVert = 3;
	else if (type == "VEC4") numPerVert = 4;
	else throw std::invalid_argument("Type is invalid (not SCALAR, VEC2, VEC3, or VEC4)");

	// Go over all the bytes in the data at the correct place using the properties from above
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
	json bufferView = JSON["bufferViews"][buffViewInd];
	unsigned int byteOffset = bufferView["byteOffset"];

	// Get indices with regards to their type: unsigned int, unsigned short, or short
	unsigned int beginningOfData = byteOffset + accByteOffset;
	if (componentType == 5125)
	{
		for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 4; i += 4)
		{
			unsigned char bytes[] = { data[i], data[i + 1], data[i + 2], data[i + 3] };
			unsigned int value;
			std::memcpy(&value, bytes, sizeof(unsigned int));
			indices.push_back((GLuint)value);
		}
	}
	else if (componentType == 5123)
	{
		for (unsigned int i = beginningOfData; i < byteOffset + accByteOffset + count * 2; i += 2)
		{
			unsigned char bytes[] = { data[i], data[i + 1] };
			unsigned short value;
			std::memcpy(&value, bytes, sizeof(unsigned short));
			indices.push_back((GLuint)value);
		}
	}
	else if (componentType == 5122)
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

    std::string fileStr = std::string(file);
    std::string fileDirectory = fileStr.substr(0, fileStr.find_last_of('/') + 1);
    AppendMessage("Model directory: " + fileDirectory);

    // Go over all images
    for (unsigned int i = 0; i < JSON["images"].size(); i++)
    {
        // uri of current texture
        std::string texPath = JSON["images"][i]["uri"];

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

			}
		);
	}
	return vertices;
}

void Model::computeTangents(std::vector<Vertex>& verts, const std::vector<GLuint>& idx)
{
    // accumulate per-vertex tangent (vec3) safely, then write final vec4 (xyz = tangent, w = handedness)
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
        // accumulate tangent (vec3)
        accumT[idx[i + 0]] += tangent;
        accumT[idx[i + 1]] += tangent;
        accumT[idx[i + 2]] += tangent;
    }

    // finalize per-vertex tangent: orthonormalize and compute handedness, then store as vec4
    for (size_t vi = 0; vi < verts.size(); ++vi)
    {
        glm::vec3 T = accumT[vi];
        glm::vec3 N = verts[vi].Normal;

        // If tangent nearly zero, provide a safe default
        if (glm::length2(T) < 1e-8f) {
            // build any perpendicular vector
            T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 0.0f, 1.0f)));
            if (glm::length2(T) < 1e-6f) // fallback
                T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 1.0f, 0.0f)));
        } else {
            // Gram-Schmidt orthogonalize tangent to normal
            T = glm::normalize(T - N * glm::dot(N, T));
        }

        glm::vec3 B = glm::cross(N, T);
        // compute handedness using original accumulated bitangent sign approximation:
        // we cannot rely on accumulated bitangent here, but following common approach use
        // sign of dot(cross(N,T), originalBitangentApprox). Use texture-space bitangent approx:
        float handedness = (glm::dot(B, glm::cross(N, T)) < 0.0f) ? -1.0f : 1.0f;
        // store tangent.xyz and handedness in w
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

void Model::CleanUp()
{
	for (auto& mesh : meshes)
	{
		mesh.CleanUp();
	}
}

std::vector<Mesh>& Model::GetMeshes()
{
	return meshes;
}

void Model::SetName(const std::string& modelName)
{
	name = modelName;
}
const std::string& Model::GetName() const
{
	return name;
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