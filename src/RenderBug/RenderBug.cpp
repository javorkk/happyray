#include "StdAfx.hpp"
#include "RenderBug/RenderBug.hpp"

#include "GLUtilities.hpp"

GLuint RenderBug::vao;
GLuint RenderBug::vbo[4];

/* Read our shaders into the appropriate buffers */
const GLchar *vertexsource_mvp = "#version 150 core\n\
                        precision highp float;\n\
                        in  vec3 in_Position; \n\
                        in  vec3 in_Color; \n\
                        in  vec3 in_Normal; \n\
                        // mvpmatrix is the result of multiplying the model, view, and projection matrices\n\
                        uniform mat4 mvpmatrix; \n\
                        // We output the ex_Color variable to the next shader in the chain \n \
                        out vec3 normal; \n\
                        out vec3 ex_Color; \n\
                        void main(void) { \n\
                            // Multiply the mvp matrix by the vertex to obtain our final vertex position\n\
                            gl_Position = mvpmatrix * vec4(in_Position, 1.0);\n\
                            normal = in_Normal;\n\
                            ex_Color = in_Color;\n\
                        }";

const GLchar *fragment_source_cartoon_shader = "#version 150 core\n\
                        // It was expressed that some drivers required this next line to function properly \n\
                        precision highp float; \n\
                        uniform vec3 lightDir;\n\
                        in  vec3 normal;\n\
                        in  vec3 ex_Color; \n\
                        out vec4 fragColor; \n\
                        \
                        void main(void) { \n\
                            float intensity; \n\
                            vec4 color; \n\
                            intensity = dot(normalize(lightDir),normalize(normal));\n\
                            if (intensity > 0.95) \n\
                               color = vec4(ex_Color * 1.2, 1.0);\n\
		                    else if (intensity > 0.25)\n\
			                    color = vec4(ex_Color * intensity, 1.0);\n\
		                    else if (intensity > 0.1)\n\
			                    color =  vec4(ex_Color * 0.1, 1.0);\n\
		                    else\n\
			                    color =  vec4(ex_Color * 0.1, 1.0);\n\
                                \
                            // Pass through our original color with full opacity. \n\
                            fragColor = color; \n\
                        }";

const GLchar *fragment_source_constant_color = "#version 150 core\n\
                        // It was expressed that some drivers required this next line to function properly \n\
                        precision highp float; \n\
                        in  vec3 normal;\n\
                        in  vec3 ex_Color; \n\
                        out vec4 fragColor; \n\
                        \
                        void main(void) { \n\
                            //float intensity; \n\
                            //vec4 color; \n\
                            //intensity = dot(normalize(lightDir),normalize(normal));\n\
							//color = vec4(ex_Color * intensity, 1.0);\n\
                            // Pass through our original color with full opacity. \n\
                            fragColor = vec4(ex_Color, 1.0); \n\
                        }";

GLuint RenderBug::vertexshader_mvp;
GLuint RenderBug::fragmentshader_cartoon;
GLuint RenderBug::fragmentshader_constant;

GLuint RenderBug::shaderprogram_cartoon;
GLuint RenderBug::shaderprogram_constant;


const GLchar *vertexsource = "#version 150 \n               \
                        precision highp float;\n                               \
                        in  vec2 in_Position; \n                               \
                        in  vec2 in_TexCoord; \n                               \
                        out vec2 aTexCoord; \n                                 \
                        void main(void) { \n                                   \
                            gl_Position = vec4(in_Position, 0.0, 1.0);\n       \
                            aTexCoord = in_TexCoord;\n                         \
                        }";
const GLchar *fragmentsource = "#version 150 \n              \
                        precision highp float; \n                              \
                        uniform sampler2D uTexture; \n                         \
                        in  vec2 aTexCoord; \n                                 \
                        out vec4 gl_FragColor; \n                              \
                                                                               \
                        vec4 gammaCorrection(float aGamma)                     \
                        {                                                      \
                        float gammaRecip = 1.0 / aGamma;                       \
                        return pow(texture2D(uTexture, aTexCoord.st),          \
                        vec4(gammaRecip, gammaRecip, gammaRecip, 1.0));        \
                        }                                                      \
                                                                               \
                        void main(void) { \n                                   \
                        gl_FragColor = gammaCorrection(2.2); \n                \
                        }";

GLuint RenderBug::vertexshader;
GLuint RenderBug::fragmentshader;
GLuint RenderBug::shaderprogram;

GLuint RenderBug::sFBTextureId;
GLuint RenderBug::sFBOId;

void RenderBug::renderScene(const CameraManager& aCamera)
{
	/* Make our background black */
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	renderTriangles(aCamera);
	renderLines(aCamera);
	renderPoints(aCamera);

	//switch (mGeometryMode)
	//{
	//case TRIANGLES:
	//	renderTriangles(aCamera);
	//	break;
	//case LINES:
	//	renderLines(aCamera);
	//	break;
	//case POINTS:
	//	renderPoints(aCamera);
	//default:
	//	renderTriangles(aCamera);
	//	break;
	//}//switch ( mGeometryMode )
}

void RenderBug::setupSceneGeometry(AnimationManager& aSceneManager)
{
	if (aSceneManager.getNumKeyFrames() < 1)
		return;

	if (numTrIndices > 0 || numLnIndices > 0 || numPtIndices > 0)
	{
		if (trIndices != NULL)
			delete[] trIndices;
		if (lnIndices != NULL)
			delete[] lnIndices;
		if (ptIndices != NULL)
			delete[] ptIndices;

		trIndices = NULL;
		lnIndices = NULL;
		ptIndices = NULL;

		numTrIndices = 0;
		numLnIndices = 0;
		numPtIndices = 0;
	}

	if (numPositions > 0)
	{
		delete[] colors;
		delete[] positions;
		delete[] normals;
		numPositions = 0u;
	}

	if (numInterpolatedPositions > 0)
	{
		delete[] interpolatedPositions;
		numInterpolatedPositions = 0u;
	}

	//The first frame
	const WFObject& keyFrame1 = aSceneManager.getFrame(aSceneManager.getFrameId());
	GeometryMode geometryModeFrame1 = keyFrame1.getNumFaces() > 0u ? TRIANGLES : keyFrame1.getNumLines() > 0u ? LINES : POINTS;
	//The second frame
	const WFObject& keyFrame2 = aSceneManager.getFrame(aSceneManager.getNextFrameId());
	GeometryMode geometryModeFrame2 = keyFrame2.getNumFaces() > 0u ? TRIANGLES : keyFrame2.getNumLines() > 0u ? LINES : POINTS;

	if (geometryModeFrame1 != geometryModeFrame2)
		return;
	mGeometryMode = geometryModeFrame1;

	//This is the interpolation coefficient. Its value is the time elapsed from frame 1
	//For example coeff == 0.5 means that we are in the middle of the 2 frames
	const float     coeff = aSceneManager.getInterpolationCoefficient();

	float4 defaultColor = make_float4(0.9f, 0.9f, 0.9f, 1.f);
	float4 color = defaultColor;

	const size_t numVertices = std::max(keyFrame1.getNumVertices(), keyFrame2.getNumVertices());
	numTrIndices = (GLsizei)std::max(keyFrame1.getNumFaces(), keyFrame2.getNumFaces());
	numLnIndices = (GLsizei)std::max(keyFrame1.getNumLines(), keyFrame2.getNumLines());
	numPtIndices = (GLsizei)std::max(keyFrame1.getNumPoints(), keyFrame2.getNumPoints());

	positions = new float[3 * (numTrIndices * 3 + numLnIndices * 2 + numPtIndices)];
	colors = new float[3 * (numTrIndices * 3 + numLnIndices * 2 + numPtIndices)];
	normals = new float[3 * (numTrIndices * 3 + numLnIndices * 2 + numPtIndices)];

	numPositions = numTrIndices * 3 + numLnIndices * 2 + numPtIndices;

	interpolatedPositions = new float[3 * numVertices];
	numInterpolatedPositions = numVertices;

	float minX = FLT_MAX;
	float minY = FLT_MAX;
	float minZ = FLT_MAX;
	float maxX = -FLT_MAX;
	float maxY = -FLT_MAX;
	float maxZ = -FLT_MAX;


	//Itereate trough the vertex array and interpolate coordinates
	size_t it = 0u;
	size_t offset = 0u;
	for (; it < std::min(keyFrame1.getNumVertices(), keyFrame2.getNumVertices()); ++it)
	{
		interpolatedPositions[3 * (offset + it) + 0] = keyFrame1.vertices[it].x * (1.f - coeff) + keyFrame2.vertices[it].x * coeff;
		interpolatedPositions[3 * (offset + it) + 1] = keyFrame1.vertices[it].y * (1.f - coeff) + keyFrame2.vertices[it].y * coeff;
		interpolatedPositions[3 * (offset + it) + 2] = keyFrame1.vertices[it].z * (1.f - coeff) + keyFrame2.vertices[it].z * coeff;

		minX = std::min(minX, interpolatedPositions[3 * (offset + it) + 0]);
		minY = std::min(minY, interpolatedPositions[3 * (offset + it) + 1]);
		minZ = std::min(minZ, interpolatedPositions[3 * (offset + it) + 2]);

		maxX = std::max(maxX, interpolatedPositions[3 * (offset + it) + 0]);
		maxY = std::max(maxY, interpolatedPositions[3 * (offset + it) + 1]);
		maxZ = std::max(maxZ, interpolatedPositions[3 * (offset + it) + 2]);

	}

	sceneDiagonalLength = sqrtf((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY) + (maxZ - minZ) * (maxZ - minZ));

	//Key frame 2 may have more vertices, add them w/o interpolation
	for (; it < keyFrame2.vertices.size(); ++it)
	{
		interpolatedPositions[3 * (offset + it) + 0] = keyFrame2.vertices[it].x;
		interpolatedPositions[3 * (offset + it) + 1] = keyFrame2.vertices[it].y;
		interpolatedPositions[3 * (offset + it) + 2] = keyFrame2.vertices[it].z;
	}

	size_t indicesOffset = 0u;

	//switch (mGeometryMode)
	//{
	//case TRIANGLES:
	trIndices = new unsigned int[numTrIndices * 3];

	//Do not interpolate  indices, only use those of the second key frame
	for (size_t faceId = 0; faceId < keyFrame2.faces.size(); ++faceId)
	{
		trIndices[3 * (indicesOffset + faceId) + 0] = (unsigned int)(offset + 3 * faceId + 0u);
		trIndices[3 * (indicesOffset + faceId) + 1] = (unsigned int)(offset + 3 * faceId + 1u);
		trIndices[3 * (indicesOffset + faceId) + 2] = (unsigned int)(offset + 3 * faceId + 2u);

		positions[3 * offset + 9 * faceId + 0] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert1 + 0];
		positions[3 * offset + 9 * faceId + 1] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert1 + 1];
		positions[3 * offset + 9 * faceId + 2] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert1 + 2];

		positions[3 * offset + 9 * faceId + 3] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert2 + 0];
		positions[3 * offset + 9 * faceId + 4] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert2 + 1];
		positions[3 * offset + 9 * faceId + 5] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert2 + 2];

		positions[3 * offset + 9 * faceId + 6] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert3 + 0];
		positions[3 * offset + 9 * faceId + 7] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert3 + 1];
		positions[3 * offset + 9 * faceId + 8] = interpolatedPositions[3 * keyFrame2.faces[faceId].vert3 + 2];

		colors[3 * offset + 9 * faceId + 0] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.x * 3.14159f;
		colors[3 * offset + 9 * faceId + 1] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.y * 3.14159f;
		colors[3 * offset + 9 * faceId + 2] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.z * 3.14159f;

		colors[3 * offset + 9 * faceId + 3] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.x * 3.14159f;
		colors[3 * offset + 9 * faceId + 4] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.y * 3.14159f;
		colors[3 * offset + 9 * faceId + 5] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.z * 3.14159f;

		colors[3 * offset + 9 * faceId + 6] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.x * 3.14159f;
		colors[3 * offset + 9 * faceId + 7] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.y * 3.14159f;
		colors[3 * offset + 9 * faceId + 8] = keyFrame2.materials[keyFrame2.faces[faceId].material].diffuseCoeff.z * 3.14159f;

		float3 normal1, normal2, normal3;
		if (faceId < keyFrame1.getNumFaces())
		{
			normal1 = ~(keyFrame1.normals[keyFrame1.faces[faceId].norm1] * (1.f - coeff) + keyFrame2.normals[keyFrame2.faces[faceId].norm1] * coeff);
			normal2 = ~(keyFrame1.normals[keyFrame1.faces[faceId].norm2] * (1.f - coeff) + keyFrame2.normals[keyFrame2.faces[faceId].norm2] * coeff);
			normal3 = ~(keyFrame1.normals[keyFrame1.faces[faceId].norm3] * (1.f - coeff) + keyFrame2.normals[keyFrame2.faces[faceId].norm3] * coeff);

		}
		else
		{
			normal1 = keyFrame2.normals[keyFrame2.faces[faceId].norm1];
			normal2 = keyFrame2.normals[keyFrame2.faces[faceId].norm2];
			normal3 = keyFrame2.normals[keyFrame2.faces[faceId].norm3];
		}

		normals[3 * offset + 9 * faceId + 0] = normal1.x;
		normals[3 * offset + 9 * faceId + 1] = normal1.y;
		normals[3 * offset + 9 * faceId + 2] = normal1.z;

		normals[3 * offset + 9 * faceId + 3] = normal2.x;
		normals[3 * offset + 9 * faceId + 4] = normal2.y;
		normals[3 * offset + 9 * faceId + 5] = normal2.z;

		normals[3 * offset + 9 * faceId + 6] = normal3.x;
		normals[3 * offset + 9 * faceId + 7] = normal3.y;
		normals[3 * offset + 9 * faceId + 8] = normal3.z;
	}

	offset += 3 * numTrIndices;
	//	break;
	//case LINES: 		
	if (numLnIndices > 0)
	{
		lnIndices = new unsigned int[numLnIndices * 2];

		//Do not interpolate  indices, only use those of the second key frame
		for (size_t lineId = 0; lineId < keyFrame2.getNumLines(); ++lineId)
		{
			lnIndices[2 * (indicesOffset + lineId) + 0] = (unsigned int)(offset + 2 * lineId + 0u);
			lnIndices[2 * (indicesOffset + lineId) + 1] = (unsigned int)(offset + 2 * lineId + 1u);

			positions[3 * offset + 6 * lineId + 0] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert1 + 0];
			positions[3 * offset + 6 * lineId + 1] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert1 + 1];
			positions[3 * offset + 6 * lineId + 2] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert1 + 2];

			positions[3 * offset + 6 * lineId + 3] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert2 + 0];
			positions[3 * offset + 6 * lineId + 4] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert2 + 1];
			positions[3 * offset + 6 * lineId + 5] = interpolatedPositions[3 * keyFrame2.lines[lineId].vert2 + 2];

			colors[3 * (keyFrame2.lines[lineId].vert1) + 0] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.x * 3.14159f;
			colors[3 * (keyFrame2.lines[lineId].vert1) + 1] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.y * 3.14159f;
			colors[3 * (keyFrame2.lines[lineId].vert1) + 2] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.z * 3.14159f;

			colors[3 * (keyFrame2.lines[lineId].vert2) + 0] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.x * 3.14159f;
			colors[3 * (keyFrame2.lines[lineId].vert2) + 1] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.y * 3.14159f;
			colors[3 * (keyFrame2.lines[lineId].vert2) + 2] = keyFrame2.materials[keyFrame2.lines[lineId].material].diffuseCoeff.z * 3.14159f;
		}
	}
	offset += 2 * numLnIndices;
	//	break;
	//case POINTS:
	if (numPtIndices > 0)
	{
		ptIndices = new unsigned int[numPtIndices * 1];

		//Do not interpolate  indices, only use those of the second key frame
		for (size_t pointId = 0; pointId < keyFrame2.getNumPoints(); ++pointId)
		{
			ptIndices[(indicesOffset + pointId)] = (unsigned int)(offset + pointId);

			positions[3 * offset + 3 * pointId + 0] = interpolatedPositions[3 * keyFrame2.points[pointId].vert1 + 0];
			positions[3 * offset + 3 * pointId + 1] = interpolatedPositions[3 * keyFrame2.points[pointId].vert1 + 1];
			positions[3 * offset + 3 * pointId + 2] = interpolatedPositions[3 * keyFrame2.points[pointId].vert1 + 2];


			colors[3 * (keyFrame2.points[pointId].vert1) + 0] = keyFrame2.materials[keyFrame2.points[pointId].material].diffuseCoeff.x * 3.14159f;
			colors[3 * (keyFrame2.points[pointId].vert1) + 1] = keyFrame2.materials[keyFrame2.points[pointId].material].diffuseCoeff.y * 3.14159f;
			colors[3 * (keyFrame2.points[pointId].vert1) + 2] = keyFrame2.materials[keyFrame2.points[pointId].material].diffuseCoeff.z * 3.14159f;
		}
	}

	//	break;
	//default:
	//	break;
	//}

}

void RenderBug::initCartoonShader()
{
	int isCompiled_VS, isCompiled_FS;
	int IsLinked;
	int maxLength;
	char *vertexInfoLog;
	char *fragmentInfoLog;
	char *shaderProgramInfoLog;

	/* Create an empty vertex shader handle */
	vertexshader_mvp = glCreateShader(GL_VERTEX_SHADER);

	/* Send the vertex shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(vertexshader_mvp, 1, (const GLchar**)&vertexsource_mvp, 0);

	/* Compile the vertex shader */
	glCompileShader(vertexshader_mvp);

	glGetShaderiv(vertexshader_mvp, GL_COMPILE_STATUS, &isCompiled_VS);
	if (isCompiled_VS == 0)
	{
		glGetShaderiv(vertexshader_mvp, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		vertexInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(vertexshader_mvp, maxLength, &maxLength, vertexInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << vertexInfoLog << "\n";

		free(vertexInfoLog);
		return;
	}

	/* Create an empty fragment shader handle */
	fragmentshader_cartoon = glCreateShader(GL_FRAGMENT_SHADER);

	/* Send the fragment shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(fragmentshader_cartoon, 1, (const GLchar**)&fragment_source_cartoon_shader, 0);

	/* Compile the fragment shader */
	glCompileShader(fragmentshader_cartoon);

	glGetShaderiv(fragmentshader_cartoon, GL_COMPILE_STATUS, &isCompiled_FS);
	if (isCompiled_FS == 0)
	{
		glGetShaderiv(fragmentshader_cartoon, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		fragmentInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(fragmentshader_cartoon, maxLength, &maxLength, fragmentInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << fragmentInfoLog << "\n";

		free(fragmentInfoLog);
		return;
	}

	/* If we reached this point it means the vertex and fragment shaders compiled and are syntax error free. */
	/* We must link them together to make a GL shader program */
	/* GL shader programs are monolithic. It is a single piece made of 1 vertex shader and 1 fragment shader. */
	/* Assign our program handle a "name" */
	shaderprogram_cartoon = glCreateProgram();

	/* Attach our shaders to our program */
	glAttachShader(shaderprogram_cartoon, vertexshader_mvp);
	glAttachShader(shaderprogram_cartoon, fragmentshader_cartoon);

	/* Bind attribute index 0 (coordinates) to in_Position and attribute index 1 (color) to in_Color */
	/* Attribute locations must be setup before calling glLinkProgram. */
	glBindAttribLocation(shaderprogram_cartoon, 0, "in_Position");
	glBindAttribLocation(shaderprogram_cartoon, 1, "in_Color");
	glBindAttribLocation(shaderprogram_cartoon, 3, "in_Normal");


	/* Link our program */
	/* At this stage, the vertex and fragment programs are inspected, optimized and a binary code is generated for the shader. */
	/* The binary code is uploaded to the GPU, if there is no error. */
	glLinkProgram(shaderprogram_cartoon);

	/* Again, we must check and make sure that it linked. If it fails, it would mean either there is a mismatch between the vertex */
	/* and fragment shaders. It might be that you have surpassed your GPU's abilities. Perhaps too many ALU operations or */
	/* too many texel fetch instructions or too many interpolators or dynamic loops. */
	glGetProgramiv(shaderprogram_cartoon, GL_LINK_STATUS, (int *)&IsLinked);
	if (IsLinked == 0)
	{
		/* Noticed that glGetProgramiv is used to get the length for a shader program, not glGetShaderiv. */
		glGetProgramiv(shaderprogram_cartoon, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		shaderProgramInfoLog = (char *)malloc(maxLength);

		/* Notice that glGetProgramInfoLog, not glGetShaderInfoLog. */
		glGetProgramInfoLog(shaderprogram_cartoon, maxLength, &maxLength, shaderProgramInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << shaderProgramInfoLog << "\n";

		/* In this simple program, we'll just leave */
		free(shaderProgramInfoLog);
		return;
	}
}

void RenderBug::initConstantShader()
{
	int isCompiled_VS, isCompiled_FS;
	int IsLinked;
	int maxLength;
	char *vertexInfoLog;
	char *fragmentInfoLog;
	char *shaderProgramInfoLog;

	/* Create an empty vertex shader handle */
	vertexshader_mvp = glCreateShader(GL_VERTEX_SHADER);

	/* Send the vertex shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(vertexshader_mvp, 1, (const GLchar**)&vertexsource_mvp, 0);

	/* Compile the vertex shader */
	glCompileShader(vertexshader_mvp);

	glGetShaderiv(vertexshader_mvp, GL_COMPILE_STATUS, &isCompiled_VS);
	if (isCompiled_VS == 0)
	{
		glGetShaderiv(vertexshader_mvp, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		vertexInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(vertexshader_mvp, maxLength, &maxLength, vertexInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << vertexInfoLog << "\n";

		free(vertexInfoLog);
		return;
	}

	/* Create an empty fragment shader handle */
	fragmentshader_constant = glCreateShader(GL_FRAGMENT_SHADER);

	/* Send the fragment shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(fragmentshader_constant, 1, (const GLchar**)&fragment_source_constant_color, 0);

	/* Compile the fragment shader */
	glCompileShader(fragmentshader_constant);

	glGetShaderiv(fragmentshader_constant, GL_COMPILE_STATUS, &isCompiled_FS);
	if (isCompiled_FS == 0)
	{
		glGetShaderiv(fragmentshader_constant, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		fragmentInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(fragmentshader_constant, maxLength, &maxLength, fragmentInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << fragmentInfoLog << "\n";

		free(fragmentInfoLog);
		return;
	}

	/* If we reached this point it means the vertex and fragment shaders compiled and are syntax error free. */
	/* We must link them together to make a GL shader program */
	/* GL shader programs are monolithic. It is a single piece made of 1 vertex shader and 1 fragment shader. */
	/* Assign our program handle a "name" */
	shaderprogram_constant = glCreateProgram();

	/* Attach our shaders to our program */
	glAttachShader(shaderprogram_constant, vertexshader_mvp);
	glAttachShader(shaderprogram_constant, fragmentshader_constant);

	/* Bind attribute index 0 (coordinates) to in_Position and attribute index 1 (color) to in_Color */
	/* Attribute locations must be setup before calling glLinkProgram. */
	glBindAttribLocation(shaderprogram_constant, 0, "in_Position");
	glBindAttribLocation(shaderprogram_constant, 1, "in_Color");


	/* Link our program */
	/* At this stage, the vertex and fragment programs are inspected, optimized and a binary code is generated for the shader. */
	/* The binary code is uploaded to the GPU, if there is no error. */
	glLinkProgram(shaderprogram_constant);

	/* Again, we must check and make sure that it linked. If it fails, it would mean either there is a mismatch between the vertex */
	/* and fragment shaders. It might be that you have surpassed your GPU's abilities. Perhaps too many ALU operations or */
	/* too many texel fetch instructions or too many interpolators or dynamic loops. */
	glGetProgramiv(shaderprogram_constant, GL_LINK_STATUS, (int *)&IsLinked);
	if (IsLinked == 0)
	{
		/* Noticed that glGetProgramiv is used to get the length for a shader program, not glGetShaderiv. */
		glGetProgramiv(shaderprogram_constant, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		shaderProgramInfoLog = (char *)malloc(maxLength);

		/* Notice that glGetProgramInfoLog, not glGetShaderInfoLog. */
		glGetProgramInfoLog(shaderprogram_constant, maxLength, &maxLength, shaderProgramInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		/* In this simple program, we'll just leave */
		free(shaderProgramInfoLog);
		return;
	}

}

void RenderBug::renderTriangles(const CameraManager& aCamera)
{
	if (numTrIndices <= 0)
		return;

	/* Allocate and assign a Vertex Array Object to our handle */
	glGenVertexArrays(1, &vao);

	/* Bind our Vertex Array Object as the current used object */
	glBindVertexArray(vao);

	/* Allocate and assign tree Vertex Buffer Objects to our handle */
	glGenBuffers(4, vbo);



	/* Load the shader into the rendering pipeline */
	glUseProgram(shaderprogram_cartoon);

	GLfloat projectionmatrix[16]; /* Our projection matrix starts with all 0s */
	GLfloat modelmatrix[16]; /* Our model matrix  */
							 /* An identity matrix we use to perform the equivalant of glLoadIdentity */
	const GLfloat identitymatrix[16] = IDENTITY_MATRIX4;

	/* Create our projection matrix with a 45 degree field of view
	* a width to height ratio of RESX/RESY and view from .1 to 100 infront of us */
	const GLfloat aspectRatio = static_cast<float>(aCamera.getResX()) / static_cast<float>(aCamera.getResY());
	perspective(projectionmatrix, aCamera.getFOV(), aspectRatio, 0.1f, sceneDiagonalLength);

	/////////////////////////////////////////////////////////////////////////////////////
	//Setup Camera and background color
	modelmatrix[0] = aCamera.getRight().x;
	modelmatrix[1] = aCamera.getUp().x;
	modelmatrix[2] = -aCamera.getOrientation().x;
	modelmatrix[3] = 0.f;

	modelmatrix[4] = aCamera.getRight().y;
	modelmatrix[5] = aCamera.getUp().y;
	modelmatrix[6] = -aCamera.getOrientation().y;
	modelmatrix[7] = 0.f;

	modelmatrix[8] = aCamera.getRight().z;
	modelmatrix[9] = aCamera.getUp().z;
	modelmatrix[10] = -aCamera.getOrientation().z;
	modelmatrix[11] = 0.f;

	modelmatrix[12] = -dot(aCamera.getPosition(), aCamera.getRight());
	modelmatrix[13] = -dot(aCamera.getPosition(), aCamera.getUp());
	modelmatrix[14] = dot(aCamera.getPosition(), aCamera.getOrientation());
	modelmatrix[15] = 1.f;

	/* multiply our modelmatrix and our projectionmatrix. Results are stored in modelmatrix */
	multiply4x4(modelmatrix, projectionmatrix);

	/* Bind our modelmatrix variable to be a uniform called mvpmatrix in our shaderprogram */
	glUniformMatrix4fv(glGetUniformLocation(shaderprogram_cartoon, "mvpmatrix"), 1, GL_FALSE, modelmatrix);

	/* Make our background black */
	//glClearColor(0.0, 0.0, 0.0, 1.0);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////////////////////////////////////////////////
	//Setup scene geometry

	const GLfloat lightDir[3] = { -1.9f, 4.2f, 4.3f };
	glUniform3f(glGetUniformLocation(shaderprogram_cartoon, "lightDir"), lightDir[0], lightDir[1], lightDir[2]);
	/////////////////////////////////////////////////////////////////////////////////////////
	//Setup OpenGL buffers

	/* Bind our first VBO as being the active buffer and storing vertex attributes (coordinates) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	/* Copy the vertex positions to our buffer */
	/* numPositions * 3 * sizeof(GLfloat) is the size of the positions array, since it contains numPositions * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)positions, GL_STATIC_DRAW);

	/* Specify that our coordinate data is going into attribute index 0, and contains tree floats per vertex */
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 0 as being used */
	glEnableVertexAttribArray(0);

	/* Bind our second VBO as being the active buffer and storing vertex attributes (colors) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	/* Copy the color data from colors to our buffer */
	/* numColors * 3 * sizeof(GLfloat) is the size of the colors array, since it contains numColors * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)colors, GL_STATIC_DRAW);

	/* Specify that our color data is going into attribute index 1, and contains three floats per vertex */
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 1 as being used */
	glEnableVertexAttribArray(1);

	/* Bind our third VBO as being the active buffer and storing vertex attributes (indices) */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);

	/* Copy the index data from indices to our buffer */
	/* numTrIndices * 3 * sizeof(GLfloat) is the size of the indices array, since it contains numTrIndices*3 GLubyte values */
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numTrIndices * 3 * sizeof(GLuint), (GLuint*)trIndices, GL_STATIC_DRAW);

	/* Specify that our index data is going into attribute index 2, and contains three ints per vertex */
	glVertexAttribPointer(2, 3, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

	/* Enable attribute index 2 as being used */
	glEnableVertexAttribArray(2);

	/* Bind our fourth VBO as being the active buffer and storing vertex attributes (normals) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);

	/* Copy the vertex normals to our buffer */
	/* numPositions * 3 * sizeof(GLfloat) is the size of the normals array, since it contains numPositions * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)normals, GL_STATIC_DRAW);

	/* Specify that our coordinate data is going into attribute index 3, and contains tree floats per vertex */
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 3 as being used */
	glEnableVertexAttribArray(3);


	/////////////////////////////////////////////////////////////////////////////////////////
	//Render scene

	/* Invoke glDrawElements telling it to draw triangles strip using indicies */
	glDrawElements(GL_TRIANGLES, numTrIndices * 3, GL_UNSIGNED_INT, 0);


	/* Disable vertex arrays */
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

	/* Cleanup all the things we bound and allocated */
	glUseProgram(0);
	glDeleteBuffers(4, vbo);
	glDeleteVertexArrays(1, &vao);
}

void RenderBug::renderLines(const CameraManager& aCamera)
{
	if (numLnIndices <= 0)
		return;

	/* Allocate and assign a Vertex Array Object to our handle */
	glGenVertexArrays(1, &vao);

	/* Bind our Vertex Array Object as the current used object */
	glBindVertexArray(vao);

	/* Allocate and assign tree Vertex Buffer Objects to our handle */
	glGenBuffers(3, vbo);


	/* Load the shader into the rendering pipeline */
	glUseProgram(shaderprogram_constant);

	GLfloat projectionmatrix[16]; /* Our projection matrix starts with all 0s */
	GLfloat modelmatrix[16]; /* Our model matrix  */
							 /* An identity matrix we use to perform the equivalant of glLoadIdentity */
	const GLfloat identitymatrix[16] = IDENTITY_MATRIX4;

	/* Create our projection matrix with a 45 degree field of view
	* a width to height ratio of RESX/RESY and view from .1 to 100 infront of us */
	const GLfloat aspectRatio = static_cast<float>(aCamera.getResX()) / static_cast<float>(aCamera.getResY());
	perspective(projectionmatrix, aCamera.getFOV(), aspectRatio, 0.1f, sceneDiagonalLength);

	/////////////////////////////////////////////////////////////////////////////////////
	//Setup Camera and background color
	modelmatrix[0] = aCamera.getRight().x;
	modelmatrix[1] = aCamera.getUp().x;
	modelmatrix[2] = -aCamera.getOrientation().x;
	modelmatrix[3] = 0.f;

	modelmatrix[4] = aCamera.getRight().y;
	modelmatrix[5] = aCamera.getUp().y;
	modelmatrix[6] = -aCamera.getOrientation().y;
	modelmatrix[7] = 0.f;

	modelmatrix[8] = aCamera.getRight().z;
	modelmatrix[9] = aCamera.getUp().z;
	modelmatrix[10] = -aCamera.getOrientation().z;
	modelmatrix[11] = 0.f;

	modelmatrix[12] = -dot(aCamera.getPosition(), aCamera.getRight());
	modelmatrix[13] = -dot(aCamera.getPosition(), aCamera.getUp());
	modelmatrix[14] = dot(aCamera.getPosition(), aCamera.getOrientation());
	modelmatrix[15] = 1.f;

	/* multiply our modelmatrix and our projectionmatrix. Results are stored in modelmatrix */
	multiply4x4(modelmatrix, projectionmatrix);

	/* Bind our modelmatrix variable to be a uniform called mvpmatrix in our shaderprogram */
	glUniformMatrix4fv(glGetUniformLocation(shaderprogram_constant, "mvpmatrix"), 1, GL_FALSE, modelmatrix);

	/* Make our background black */
	//glClearColor(0.0, 0.0, 0.0, 1.0);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////////////////////////////////////////////////
	//Setup OpenGL buffers

	/* Bind our first VBO as being the active buffer and storing vertex attributes (coordinates) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	/* Copy the vertex positions to our buffer */
	/* numPositions * 3 * sizeof(GLfloat) is the size of the positions array, since it contains numPositions * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)positions, GL_STATIC_DRAW);

	/* Specify that our coordinate data is going into attribute index 0, and contains tree floats per vertex */
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 0 as being used */
	glEnableVertexAttribArray(0);

	/* Bind our second VBO as being the active buffer and storing vertex attributes (colors) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	/* Copy the color data from colors to our buffer */
	/* numColors * 3 * sizeof(GLfloat) is the size of the colors array, since it contains numColors * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)colors, GL_STATIC_DRAW);

	/* Specify that our color data is going into attribute index 1, and contains three floats per vertex */
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 1 as being used */
	glEnableVertexAttribArray(1);

	/* Bind our third VBO as being the active buffer and storing vertex attributes (indices) */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);

	/* Copy the index data from indices to our buffer */
	/* numIndices * 2 * sizeof(GLuint) is the size of the indices array, since it contains numIndices*2 GLubyte values */
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numLnIndices * 2 * sizeof(GLuint), (GLuint*)lnIndices, GL_STATIC_DRAW);

	/* Specify that our index data is going into attribute index 2, and contains three ints per vertex */
	glVertexAttribPointer(2, 2, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

	/* Enable attribute index 2 as being used */
	glEnableVertexAttribArray(2);

	/////////////////////////////////////////////////////////////////////////////////////////
	//Render scene

	/* Invoke glDrawElements telling it to draw lines using indicies */
	glDrawElements(GL_LINES, numLnIndices * 2, GL_UNSIGNED_INT, 0);


	/* Disable vertex arrays */
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);


	/* Cleanup all the things we bound and allocated */
	glUseProgram(0);
	glDeleteBuffers(3, vbo);
	glDeleteVertexArrays(1, &vao);
}

void RenderBug::renderPoints(const CameraManager& aCamera)
{
	if (numPtIndices <= 0)
		return;

	/* Allocate and assign a Vertex Array Object to our handle */
	glGenVertexArrays(1, &vao);

	/* Bind our Vertex Array Object as the current used object */
	glBindVertexArray(vao);

	/* Allocate and assign tree Vertex Buffer Objects to our handle */
	glGenBuffers(3, vbo);



	/* Load the shader into the rendering pipeline */
	glUseProgram(shaderprogram_constant);

	GLfloat projectionmatrix[16]; /* Our projection matrix starts with all 0s */
	GLfloat modelmatrix[16]; /* Our model matrix  */
							 /* An identity matrix we use to perform the equivalant of glLoadIdentity */
	const GLfloat identitymatrix[16] = IDENTITY_MATRIX4;

	/* Create our projection matrix with a 45 degree field of view
	* a width to height ratio of RESX/RESY and view from .1 to 100 infront of us */
	const GLfloat aspectRatio = static_cast<float>(aCamera.getResX()) / static_cast<float>(aCamera.getResY());
	perspective(projectionmatrix, aCamera.getFOV(), aspectRatio, 0.1f, sceneDiagonalLength);

	/////////////////////////////////////////////////////////////////////////////////////
	//Setup Camera and background color
	modelmatrix[0] = aCamera.getRight().x;
	modelmatrix[1] = aCamera.getUp().x;
	modelmatrix[2] = -aCamera.getOrientation().x;
	modelmatrix[3] = 0.f;

	modelmatrix[4] = aCamera.getRight().y;
	modelmatrix[5] = aCamera.getUp().y;
	modelmatrix[6] = -aCamera.getOrientation().y;
	modelmatrix[7] = 0.f;

	modelmatrix[8] = aCamera.getRight().z;
	modelmatrix[9] = aCamera.getUp().z;
	modelmatrix[10] = -aCamera.getOrientation().z;
	modelmatrix[11] = 0.f;

	modelmatrix[12] = -dot(aCamera.getPosition(), aCamera.getRight());
	modelmatrix[13] = -dot(aCamera.getPosition(), aCamera.getUp());
	modelmatrix[14] = dot(aCamera.getPosition(), aCamera.getOrientation());
	modelmatrix[15] = 1.f;

	/* multiply our modelmatrix and our projectionmatrix. Results are stored in modelmatrix */
	multiply4x4(modelmatrix, projectionmatrix);

	/* Bind our modelmatrix variable to be a uniform called mvpmatrix in our shaderprogram */
	glUniformMatrix4fv(glGetUniformLocation(shaderprogram_constant, "mvpmatrix"), 1, GL_FALSE, modelmatrix);

	/////////////////////////////////////////////////////////////////////////////////////////
	//Setup OpenGL buffers

	/* Bind our first VBO as being the active buffer and storing vertex attributes (coordinates) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	/* Copy the vertex positions to our buffer */
	/* numPositions * 3 * sizeof(GLfloat) is the size of the positions array, since it contains numPositions * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)positions, GL_STATIC_DRAW);

	/* Specify that our coordinate data is going into attribute index 0, and contains tree floats per vertex */
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 0 as being used */
	glEnableVertexAttribArray(0);

	/* Bind our second VBO as being the active buffer and storing vertex attributes (colors) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	/* Copy the color data from colors to our buffer */
	/* numColors * 3 * sizeof(GLfloat) is the size of the colors array, since it contains numColors * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, numPositions * 3 * sizeof(GLfloat), (GLfloat*)colors, GL_STATIC_DRAW);

	/* Specify that our color data is going into attribute index 1, and contains three floats per vertex */
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 1 as being used */
	glEnableVertexAttribArray(1);

	/* Bind our third VBO as being the active buffer and storing vertex attributes (indices) */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);

	/* Copy the index data from indices to our buffer */
	/* numIndices * 1 * sizeof(GLfloat) is the size of the indices array, since it contains numIndices*1 GLubyte values */
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numPtIndices * 1 * sizeof(GLuint), (GLuint*)ptIndices, GL_STATIC_DRAW);

	/* Specify that our index data is going into attribute index 2, and contains one int per vertex */
	glVertexAttribPointer(2, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

	/* Enable attribute index 2 as being used */
	glEnableVertexAttribArray(2);


	/////////////////////////////////////////////////////////////////////////////////////////
	//Render scene

	/* Invoke glDrawElements telling it to draw triangles strip using indicies */
	glDrawElements(GL_POINTS, numPtIndices * 1, GL_UNSIGNED_INT, 0);

	/* Disable vertex arrays */
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	/* Cleanup all the things we bound and allocated */
	glUseProgram(0);
	glDeleteBuffers(3, vbo);
	glDeleteVertexArrays(1, &vao);
}

void RenderBug::cleanup()
{
	//texture
	glDeleteTextures(1, &sFBTextureId);
	//shader & buffers
	glUseProgram(0);
	glDetachShader(shaderprogram_cartoon, vertexshader_mvp);
	glDetachShader(shaderprogram_cartoon, fragmentshader_cartoon);
	glDeleteProgram(shaderprogram_cartoon);

	glDetachShader(shaderprogram_constant, vertexshader_mvp);
	glDetachShader(shaderprogram_constant, fragmentshader_constant);
	glDeleteProgram(shaderprogram_constant);

	glDeleteShader(vertexshader_mvp);
	glDeleteShader(fragmentshader_cartoon);
	glDeleteShader(fragmentshader_constant);


	glDetachShader(shaderprogram, vertexshader);
	glDetachShader(shaderprogram, fragmentshader);
	glDeleteProgram(shaderprogram);

	glDeleteShader(vertexshader);
	glDeleteShader(fragmentshader);

	glDeleteBuffers(4, vbo);
	glDeleteVertexArrays(1, &vao);
}


void RenderBug::initFBufferShader()
{
	int isCompiled_VS, isCompiled_FS;
	int IsLinked;
	int maxLength;
	char *vertexInfoLog;
	char *fragmentInfoLog;
	char *shaderProgramInfoLog;

	/* Allocate and assign a Vertex Array Object to our handle */
	glGenVertexArrays(1, &vao);

	/* Bind our Vertex Array Object as the current used object */
	glBindVertexArray(vao);

	/* Allocate and assign tree Vertex Buffer Objects to our handle */
	glGenBuffers(3, vbo);

	/* Create an empty vertex shader handle */
	vertexshader = glCreateShader(GL_VERTEX_SHADER);

	/* Send the vertex shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(vertexshader, 1, (const GLchar**)&vertexsource, 0);

	/* Compile the vertex shader */
	glCompileShader(vertexshader);

	glGetShaderiv(vertexshader, GL_COMPILE_STATUS, &isCompiled_VS);
	if (isCompiled_VS == 0)
	{
		glGetShaderiv(vertexshader, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		vertexInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(vertexshader, maxLength, &maxLength, vertexInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << vertexInfoLog << "\n";

		free(vertexInfoLog);
		return;
	}

	/* Create an empty fragment shader handle */
	fragmentshader = glCreateShader(GL_FRAGMENT_SHADER);

	/* Send the fragment shader source code to GL */
	/* Note that the source code is NULL character terminated. */
	/* GL will automatically detect that therefore the length info can be 0 in this case (the last parameter) */
	glShaderSource(fragmentshader, 1, (const GLchar**)&fragmentsource, 0);

	/* Compile the fragment shader */
	glCompileShader(fragmentshader);

	glGetShaderiv(fragmentshader, GL_COMPILE_STATUS, &isCompiled_FS);
	if (isCompiled_FS == 0)
	{
		glGetShaderiv(fragmentshader, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		fragmentInfoLog = (char *)malloc(maxLength);

		glGetShaderInfoLog(fragmentshader, maxLength, &maxLength, fragmentInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		std::cerr << "Shader program compilation failed\n";
		std::cerr << fragmentInfoLog << "\n";

		free(fragmentInfoLog);
		return;
	}

	/* If we reached this point it means the vertex and fragment shaders compiled and are syntax error free. */
	/* We must link them together to make a GL shader program */
	/* GL shader programs are monolithic. It is a single piece made of 1 vertex shader and 1 fragment shader. */
	/* Assign our program handle a "name" */
	shaderprogram = glCreateProgram();

	/* Attach our shaders to our program */
	glAttachShader(shaderprogram, vertexshader);
	glAttachShader(shaderprogram, fragmentshader);

	/* Bind attribute index 0 (coordinates) to in_Position, attribute index 1 to in_TexCoord */
	/* Attribute locations must be setup before calling glLinkProgram. */
	glBindAttribLocation(shaderprogram, 0, "in_Position");
	glBindAttribLocation(shaderprogram, 1, "in_TexCoord");

	/* Link our program */
	/* At this stage, the vertex and fragment programs are inspected, optimized and a binary code is generated for the shader. */
	/* The binary code is uploaded to the GPU, if there is no error. */
	glLinkProgram(shaderprogram);

	/* Again, we must check and make sure that it linked. If it fails, it would mean either there is a mismatch between the vertex */
	/* and fragment shaders. It might be that you have surpassed your GPU's abilities. Perhaps too many ALU operations or */
	/* too many texel fetch instructions or too many interpolators or dynamic loops. */
	glGetProgramiv(shaderprogram, GL_LINK_STATUS, (int *)&IsLinked);
	if (IsLinked == 0)
	{
		/* Noticed that glGetProgramiv is used to get the length for a shader program, not glGetShaderiv. */
		glGetProgramiv(shaderprogram, GL_INFO_LOG_LENGTH, &maxLength);

		/* The maxLength includes the NULL character */
		shaderProgramInfoLog = (char *)malloc(maxLength);

		/* Notice that glGetProgramInfoLog, not glGetShaderInfoLog. */
		glGetProgramInfoLog(shaderprogram, maxLength, &maxLength, shaderProgramInfoLog);

		/* Handle the error in an appropriate way such as displaying a message or writing to a log file. */
		/* In this simple program, we'll just leave */
		free(shaderProgramInfoLog);
		return;
	}
}

void RenderBug::initFrameBufferTexture(const int aResX, const int aResY)
{
	// create a new texture name
	glGenTextures(1, &sFBTextureId);
	// bind the texture name to a texture target
	glBindTexture(TEXTURE_TARGET, sFBTextureId);
	// turn off filtering and set proper wrap mode 
	// (obligatory for float textures atm)
	glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(TEXTURE_TARGET, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	// set texenv to replace instead of the default modulate
	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	// and allocate graphics memory

	glTexImage2D(TEXTURE_TARGET,
		0, //not to use any mipmap levels for this texture
		INTERNAL_FORMAT,
		aResX,
		aResY,
		0, //turns off borders
		TEXTURE_FORMAT,
		GL_FLOAT,
		0);
}

void RenderBug::renderFBuffer(float*& aFrameBufferFloatPtr, const int aResX, const int aResY)
{
	/* Load the shader into the rendering pipeline */
	glUseProgram(shaderprogram);

	glEnable(TEXTURE_TARGET);
	// enable texture x (read-only, not changed in the computation loop)
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(TEXTURE_TARGET, sFBTextureId);
	glUniform1i(glGetUniformLocation(shaderprogram, "uTexture"), 0); // texture unit 0
																	 //////////////////////////////////////////////////////////////////////////
																	 //NVidia
	glTexSubImage2D(TEXTURE_TARGET, 0, 0, 0, aResX, aResY, TEXTURE_FORMAT, GL_FLOAT, aFrameBufferFloatPtr);
	//////////////////////////////////////////////////////////////////////////

	const GLfloat quad[4][2] =
	{
		{ -1.f, -1.f },
		{ 1.f, -1.f },
		{ 1.f, 1.f },
		{ -1.f, 1.f }
	};

	const GLfloat texCoords[4][2] =
	{
		{ 0.f, 1.f },
		{ 1.f, 1.f },
		{ 1.f, 0.f },
		{ 0.f, 0.f }
	};

	const GLuint indices[6] = { 0,1,2,0,3,2 };


	/////////////////////////////////////////////////////////////////////////////////////////
	//Setup OpenGL buffers

	/* Bind our first VBO as being the active buffer and storing vertex attributes (coordinates) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	/* Copy the vertex positions to our buffer */
	/* 4 * 3 * sizeof(GLfloat) is the size of the positions array, since it contains 4 * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), quad, GL_STATIC_DRAW);

	/* Specify that our coordinate data is going into attribute index 0, and contains 2 floats per vertex */
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 0 as being used */
	glEnableVertexAttribArray(0);

	/* Bind our second VBO as being the active buffer and storing vertex attributes (texture coordinates) */
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	/* Copy the color data from colors to our buffer */
	/* 4 * 3 * sizeof(GLfloat) is the size of the colors array, since it contains 4 * 3 GLfloat values */
	glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);

	/* Specify that our color data is going into attribute index 1, and contains 2 floats per vertex */
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

	/* Enable attribute index 1 as being used */
	glEnableVertexAttribArray(1);

	/* Bind our third VBO as being the active buffer and storing vertex attributes (indices) */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[2]);

	/* Copy the index data from indices to our buffer */
	/* 2 * 3 * sizeof(GLfloat) is the size of the indices array, since it contains 2*3 GLubyte values */
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 3 * sizeof(GLuint), indices, GL_STATIC_DRAW);

	/* Specify that our index data is going into attribute index 2, and contains three ints per vertex */
	glVertexAttribPointer(2, 3, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

	/* Enable attribute index 2 as being used */
	glEnableVertexAttribArray(2);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Invoke glDrawElements telling it to draw triangles using indicies */
	glDrawElements(GL_TRIANGLES, 2 * 3, GL_UNSIGNED_INT, 0);

	/* Disable vertex arrays */
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	//un-bind texture
	glBindTexture(TEXTURE_TARGET, 0);

	glDisable(TEXTURE_TARGET);

	/* Unload the shader from the rendering pipeline */
	glUseProgram(0);
}