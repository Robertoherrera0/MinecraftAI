# MinecraftAI
Capstone project 

**Make sure to put everything besides python files in the .gitignore (in case you have different files than the ones already in there). 

# installing 0.3.7
- you should be able to automatically install all of the requirements using requirements.txt in the command line
- also, other than those requirements the only thing I did was edit the first

```python
 repositories {
        maven { url "https://repo.spongepowered.org/maven/" }
        mavenCentral()
        maven {
            url = "https://files.minecraftforge.net/maven"
        }
        maven {
            name = "sonatype"
            url = "https://oss.sonatype.org/content/repositories/snapshots/"
        }
    }
    dependencies {
        classpath 'org.ow2.asm:asm:6.0'
        classpath 'org.spongepowered:mixingradle:0.6-SNAPSHOT' // Or latest 0.6 version
        classpath 'net.minecraftforge.gradle:ForgeGradle:2.2-SNAPSHOT'
    }
```
